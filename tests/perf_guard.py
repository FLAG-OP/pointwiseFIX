"""性能护栏: 广播 add 修复前后对照。

修复不改变默认路径 (Ascend 默认 bptr=False, 广播守卫是 no-op),
但 bptr 环境下广播从"崩"变"走线性寻址", 应与非广播线性路径同量级。
"""
import statistics
import time

import torch
import torch_npu
import flag_gems
from flag_gems.utils.codegen_config_utils import get_codegen_config

flag_gems.enable()


def bench(fn, iters=100, warmup=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(ts)


A = torch.randn(4096, 1, device="npu:0")
B = torch.randn(1, 4096, device="npu:0")
M1 = torch.randn(4096, 4096, device="npu:0")
M2 = torch.randn(4096, 4096, device="npu:0")

# 默认配置 (bptr=False): 修复是 no-op, 前后应一致 —— 只测当前值作基线
t_bcast = bench(lambda: torch.add(A, B))
t_same = bench(lambda: torch.add(M1, M2))
print(f"[默认配置(bptr=False), 修复为 no-op]")
print(f"  广播 add (4096,1)+(1,4096):  {t_bcast:8.1f} us")
print(f"  同形 add (4096,4096):        {t_same:8.1f} us  (参照)")

# bptr 环境模拟: 广播走线性寻址 (修复后), 非广播走 bptr
cfg = get_codegen_config()
cfg.prefer_1d_tile = False
cfg.prefer_block_pointer = True
t_bcast2 = bench(lambda: torch.add(A, B))
t_same2 = bench(lambda: torch.add(M1, M2))
print(f"[bptr 强制开启环境]")
print(f"  广播 add (守卫->线性寻址):   {t_bcast2:8.1f} us  (修复前: 编译崩)")
print(f"  同形 add (bptr 路径):        {t_same2:8.1f} us")
print(f"  广播 vs 同形 比值: {t_bcast2/t_same2:.2f} (合理范围 ~1x, 广播少读内存甚至可更快)")
