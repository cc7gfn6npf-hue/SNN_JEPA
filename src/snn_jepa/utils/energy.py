"""能量分析：统计脉冲发放率，估算 SynOps（突触操作数）。

发放率是 SNN 能耗的核心代理：发放率越低，突触操作（SynOps）越少，能耗越低。
等价的 ANN 每层发放率恒为 1（所有神经元都执行乘加 MAC）。
"""

import torch


def collect_firing_rates(encoder, loader, device, max_batches=10):
    """统计编码器各脉冲层的平均发放率（spike 稀疏性）。

    Args:
        encoder: SpikingEncoder。
        loader: 数据加载器。
        device: 设备。
        max_batches: 统计的最大批次数。

    Returns:
        dict: 层名 -> 平均发放率（0~1）。
    """
    encoder.eval()
    stats = {}
    hooks = []

    def make_collector(d):
        def hook(module, inp, out):
            d["count"] += out.sum().item()
            d["total"] += out.numel()

        return hook

    for name, module in encoder.named_modules():
        if module.__class__.__name__ in ("IFNode", "LIFNode", "ParametricLIFNode"):
            d = {"count": 0.0, "total": 0.0}
            stats[name] = d
            hooks.append(module.register_forward_hook(make_collector(d)))

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            encoder(x.to(device))

    for h in hooks:
        h.remove()

    rates = {name: d["count"] / max(d["total"], 1.0) for name, d in stats.items()}
    return rates
