"""能量分析脚本：统计 SNN 编码器的发放率与相对 ANN 的能耗节省。

用法：
    python scripts/energy_analysis.py --ckpt experiments/jepa_cifar10_seed42.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from snn_jepa.data.datasets import get_transforms
from snn_jepa.models.encoder import SpikingEncoder
from snn_jepa.utils.energy import collect_firing_rates
from snn_jepa.utils.seed import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="cifar10", choices=["cifar10", "stl10"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--timesteps", type=int, default=4)
    p.add_argument("--neuron", default="IF", choices=["IF", "LIF", "PLIF"])
    p.add_argument("--v-threshold", type=float, default=0.5)
    p.add_argument("--readout", default="spike_count", choices=["spike_count", "membrane"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 32 if args.data == "cifar10" else 96

    eval_tf = get_transforms(args.data, image_size, train=False)
    if args.data == "cifar10":
        test_ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=eval_tf)
    else:
        test_ds = datasets.STL10(args.data_dir, split="test", download=True, transform=eval_tf)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    encoder = SpikingEncoder(
        in_channels=3, channels=[32, 64, 128], embedding_dim=args.embed_dim,
        timesteps=args.timesteps, neuron_type=args.neuron, v_threshold=args.v_threshold,
        readout=args.readout,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    rates = collect_firing_rates(encoder, loader, device)
    print("=== 各层发放率（spike 稀疏性；ANN 等价物恒为 1.0）===")
    for name, r in sorted(rates.items()):
        print(f"  {name}: {r:.4f}")
    if rates:
        avg = sum(rates.values()) / len(rates)
        print(f"\n平均发放率: {avg:.4f}")
        print(f"相对 ANN 的突触操作节省（粗略）: {(1 - avg) * 100:.1f}%")


if __name__ == "__main__":
    main()
