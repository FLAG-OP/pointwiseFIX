"""强制 block pointer 复现实验: ASCEND config 的 prefer_block_pointer 强制 True,
跑 issue 原例, 看是否触发 'strides must not be zero'。

原理: block pointer 路径 (make_block_ptr / make_tensor_ptr) 无法表示 0 stride
(NVIDIA 侧同 bug 的报错是 'Last dimension must be contiguous', PR #5668 记录)。
issue 环境的老 triton-ascend (flagtree 0.6.0/ascend3.2, triton 3.2) 可能
默认走了该路径, 或 pass 对 0 stride 常量敏感 (ConvertLinalgRToBinary)。
"""
import torch
import torch_npu
import triton
import flag_gems
from flag_gems.utils.codegen_config_utils import get_codegen_config

print("triton:", triton.__version__)
cfg = get_codegen_config()
print("before: 1d_tile =", cfg.prefer_1d_tile, "bptr =", cfg.prefer_block_pointer)

# 强制走 block pointer 路径 (模拟 NVIDIA 默认 / 老 ascend 行为)
cfg.prefer_1d_tile = False
cfg.prefer_block_pointer = True

flag_gems.enable()
a = torch.randn(128, 1, device="npu:0")
b = torch.randn(1, 128, device="npu:0")
ref = torch.add(a.cpu(), b.cpu())

try:
    c = torch.add(a, b)
    torch.npu.synchronize()
    print("bptr add:", tuple(c.shape), "allclose:", torch.allclose(c.cpu(), ref))
except Exception as e:
    print(f"BPTR ADD FAILED: {type(e).__name__}: {str(e)[:200]}")
