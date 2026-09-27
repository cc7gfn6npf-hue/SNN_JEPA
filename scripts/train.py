"""训练脚本：SNN-JEPA 与 SNN-MAE 基线。

用法：
    python scripts/train.py --target jepa --data cifar10 --epochs 100
    python scripts/train.py --target mae  --data cifar10 --epochs 100

说明：阶段 1 MVP 用 argparse（快速跑通），configs/ 已就绪，稳定后迁移 Hydra。
"""

import argparse
import os

import torch
import torch.nn.utils as clip

from snn_jepa import losses
from snn_jepa.data.datasets import get_dataloaders
from snn_jepa.models.ann_encoder import ANNEncoder
from snn_jepa.models.decoder import Decoder
from snn_jepa.models.encoder import SpikingEncoder
from snn_jepa.models.predictor import Predictor
from snn_jepa.utils.metrics import effective_rank, stable_rank
from snn_jepa.utils.seed import set_seed


def compute_rank(encoder, loader, device):
    """用测试集一个 batch 计算表征的有效秩与稳定秩。"""
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            z = encoder(x.to(device))
            return float(effective_rank(z)), float(stable_rank(z))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="cifar10", choices=["cifar10", "stl10"])
    p.add_argument("--target", default="jepa", choices=["jepa", "mae"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--timesteps", type=int, default=4)
    p.add_argument("--neuron", default="IF", choices=["IF", "LIF", "PLIF"])
    p.add_argument("--v-threshold", type=float, default=0.5)
    p.add_argument("--readout", default="spike_count", choices=["spike_count", "membrane"])
    p.add_argument("--model", default="snn", choices=["snn", "ann"])
    p.add_argument("--lambda-std", type=float, default=1.0)
    p.add_argument("--lambda-cov", type=float, default=25.0)
    p.add_argument("--regularizer", default="vicreg", choices=["vicreg", "sigreg", "bio", "rate", "wsigreg"])
    p.add_argument("--lambda-reg", type=float, default=10.0, help="SIGReg 权衡超参")
    p.add_argument("--num-directions", type=int, default=8, help="SIGReg 随机投影方向数")
    p.add_argument("--lambda-homeo", type=float, default=10.0, help="发放率稳态权重（bio 正则）")
    p.add_argument("--target-rate", type=float, default=0.25, help="发放率稳态目标发放率（bio 正则）")
    p.add_argument("--refractory-steps", type=int, default=-1,
                   help="不应期梯度门控步数：-1=硬stop-gradient，0=无门控，>0=前N步梯度阻断（H4'）")
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./experiments")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 32 if args.data == "cifar10" else 96

    train_loader, test_loader = get_dataloaders(
        name=args.data, data_dir=args.data_dir, image_size=image_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        target=args.target, mask_ratio=args.mask_ratio,
    )

    if args.model == "ann":
        encoder = ANNEncoder(
            in_channels=3, channels=[32, 64, 128], embedding_dim=args.embed_dim,
        ).to(device)
    else:
        encoder = SpikingEncoder(
            in_channels=3, channels=[32, 64, 128], embedding_dim=args.embed_dim,
            timesteps=args.timesteps, neuron_type=args.neuron, v_threshold=args.v_threshold,
            readout=args.readout,
        ).to(device)

    if args.target == "jepa":
        head = Predictor(args.embed_dim).to(device)
    else:
        head = Decoder(args.embed_dim, in_channels=3, init_resolution=image_size // 4).to(device)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.target}_{args.data}_seed{args.seed}.pt")

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        total_loss = total_pred = total_reg = 0.0
        n_batches = 0
        for batch in train_loader:
            if args.target == "jepa":
                x1, x2 = batch
                x1, x2 = x1.to(device), x2.to(device)
                z1 = encoder(x1)
                if args.model == "ann" or args.refractory_steps < 0:
                    with torch.no_grad():
                        z2 = encoder(x2)  # 硬 stop-gradient
                else:
                    z2 = encoder(x2, detach_steps=args.refractory_steps)  # 不应期梯度门控
                pred = head(z1)
                if args.regularizer == "sigreg":
                    loss, pred_loss, reg = losses.jepa_sigreg_loss(
                        z1, z2, pred, args.lambda_reg, args.num_directions
                    )
                elif args.regularizer == "bio":
                    loss, pred_loss, reg = losses.jepa_bio_loss(
                        z1, z2, pred, args.lambda_homeo, args.lambda_cov,
                        args.target_rate, args.timesteps,
                    )
                elif args.regularizer == "rate":
                    loss, pred_loss, reg = losses.jepa_rate_loss(
                        z1, z2, pred, args.lambda_reg,
                        args.target_rate, args.timesteps, args.num_directions,
                    )
                elif args.regularizer == "wsigreg":
                    loss, pred_loss, reg = losses.jepa_weak_sigreg_loss(
                        z1, z2, pred, args.lambda_reg, args.num_directions
                    )
                else:
                    loss, pred_loss, reg = losses.jepa_loss(
                        z1, z2, pred, args.lambda_std, args.lambda_cov
                    )
            else:
                masked, x = batch
                masked, x = masked.to(device), x.to(device)
                z = encoder(masked)
                recon = head(z)
                loss = losses.mae_loss(recon, x)
                pred_loss, reg = loss, torch.zeros(1, device=device)

            optimizer.zero_grad()
            loss.backward()
            clip.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_pred += pred_loss.item()
            total_reg += reg.item()
            n_batches += 1

        erank, srank = compute_rank(encoder, test_loader, device)
        print(
            f"[epoch {epoch:3d}/{args.epochs}] loss={total_loss / n_batches:.4f} "
            f"pred={total_pred / n_batches:.4f} reg={total_reg / n_batches:.4f} "
            f"eff_rank={erank:.1f} srank={srank:.1f}"
        )

    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
