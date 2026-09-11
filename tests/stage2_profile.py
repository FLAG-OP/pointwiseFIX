"""框架层 89.7us 的内部分解: cProfile flag_gems add 调用 (排除 triton launch)。"""
import cProfile
import io
import pstats

import torch
import torch_npu
import flag_gems

flag_gems.enable()
n = 1024 * 1024
a = torch.randn(n, device="npu:0")
b = torch.randn(n, device="npu:0")

for _ in range(50):
    torch.add(a, b)
torch.npu.synchronize()

pr = cProfile.Profile()
pr.enable()
for _ in range(2000):
    torch.add(a, b)
pr.disable()
torch.npu.synchronize()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(24)
out = s.getvalue()
# 只打印 flag_gems/triton 相关的关键行
for line in out.splitlines():
    if any(k in line for k in ("pointwise", "prepare_args", "instantiate", "jit.py", "launch", "StridedBuffer", "wrapper", "broadcast", "type_promotion", "ncalls", "function calls")):
        print(line[:150])
