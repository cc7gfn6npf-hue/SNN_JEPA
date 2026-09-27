"""可复现性：固定全局随机种子。"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch（CPU 与 CUDA）的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 完全确定模式（会牺牲部分速度，换取 bit 级可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """数据加载子进程的确定性初始化。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
