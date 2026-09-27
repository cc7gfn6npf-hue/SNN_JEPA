"""线性探测：冻结编码器，训练线性分类头评估表征质量。

用法：
    python scripts/linear_probe.py --ckpt experiments/jepa_cifar10_seed42.pt
"""

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets

from snn_jepa.data.datasets import get_transforms
from snn_jepa.models.encoder import SpikingEncoder
from snn_jepa.utils.metrics import effective_rank, stable_rank
from snn_jepa.utils.seed import set_seed


def extract(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            z = encoder(x.to(device))
            feats.append(z.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="cifar10", choices=["cifar10", "stl10"])
    p.add_argument("--ckpt", required=True, help="训练好的 checkpoint 路径")
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--timesteps", type=int, default=4)
    p.add_argument("--neuron", default="IF", choices=["IF", "LIF", "PLIF"])
    p.add_argument("--v-threshold", type=float, default=0.5)
    p.add_argument("--readout", default="spike_count", choices=["spike_count", "membrane"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0, help="LogisticRegression 正则强度倒数")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 32 if args.data == "cifar10" else 96

    eval_tf = get_transforms(args.data, image_size, train=False)
    if args.data == "cifar10":
        train_ds = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=eval_tf)
        test_ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=eval_tf)
    else:
        train_ds = datasets.STL10(args.data_dir, split="train", download=True, transform=eval_tf)
        test_ds = datasets.STL10(args.data_dir, split="test", download=True, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    encoder = SpikingEncoder(
        in_channels=3, channels=[32, 64, 128], embedding_dim=args.embed_dim,
        timesteps=args.timesteps, neuron_type=args.neuron, v_threshold=args.v_threshold,
        readout=args.readout,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    X_train, y_train = extract(encoder, train_loader, device)
    X_test, y_test = extract(encoder, test_loader, device)

    # 在标准化前保存原始特征，用于表征质量指标（避免标准化掩盖维度坍塌）
    Z_raw = torch.from_numpy(X_test)

    # 标准化特征，加速 LogisticRegression 收敛（脉冲计数特征尺度不均）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=5000, C=args.C)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    print(f"linear probe accuracy: {acc * 100:.2f}%")
    print(f"effective rank: {effective_rank(Z_raw):.1f}  |  stable rank: {stable_rank(Z_raw):.1f}")
    print(f"max rank (min(N, D)): {min(X_test.shape[0], X_test.shape[1])}")


if __name__ == "__main__":
    main()
