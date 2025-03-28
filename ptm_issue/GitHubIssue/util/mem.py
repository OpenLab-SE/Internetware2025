import os
import torch


def check_mem(cuda_device):
    """
    获取当前显存使用情况
    """
    devices_info = os.popen(
        '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used


def occupy_mem(cuda_device):
    """
    分配未使用显存
    """
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.8)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem, device=f'cuda:{cuda_device}')
    del x
