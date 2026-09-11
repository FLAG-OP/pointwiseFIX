"""隔离实验: bptr 模式下, 广播 vs 非广播。
若非广播 OK + 广播崩 -> 0 stride 是唯一变量, 根因闭环。"""
import torch
import torch_npu
import flag_gems
from flag_gems.utils.codegen_config_utils import get_codegen_config

cfg = get_codegen_config()
cfg.prefer_1d_tile = False
cfg.prefer_block_pointer = True

flag_gems.enable()

a = torch.randn(128, 128, device="npu:0")   # 同形, 无广播
b = torch.randn(128, 128, device="npu:0")
try:
    c = torch.add(a, b)
    torch.npu.synchronize()
    print("[bptr 非广播 (128,128)+(128,128)] OK allclose=",
          torch.allclose(c.cpu(), a.cpu() + b.cpu()))
except Exception as e:
    print(f"[bptr 非广播] FAILED {type(e).__name__}: {str(e)[:80]}")

a2 = torch.randn(128, 1, device="npu:0")    # 广播
b2 = torch.randn(1, 128, device="npu:0")
try:
    c2 = torch.add(a2, b2)
    torch.npu.synchronize()
    print("[bptr 广播 (128,1)+(1,128)] OK")
except Exception as e:
    print(f"[bptr 广播 (128,1)+(1,128)] FAILED {type(e).__name__}: {str(e)[:80]}")
