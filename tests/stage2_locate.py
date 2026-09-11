"""stage-2 定位测量: host enqueue 开销三层分解。

层1: 原生 torch.add        —— C++ dispatch + aclnn enqueue (理论上限)
层2: 裸 triton kernel      —— triton JITFunction.__call__ 的 Python launch 链
层3: flag_gems add (codegen) —— torch dispatch → prepare_args → 生成 wrapper → triton

测法: host-only 计时 (sync 后起表, N 次 enqueue, 停表, 再 sync)。
stage-2 的目标数字 = 层3 - 层1; 其中 triton 链路 (层2) 与框架层 (层3-层2) 各占多少。
"""
import statistics
import time

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def add_flat(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)


def host_time(fn, iters=500):
    """host-only enqueue 时间: 排队不等待完成。"""
    for _ in range(50):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        t1 = time.perf_counter()
        torch.npu.synchronize()
        ts.append((t1 - t0) / iters * 1e6)
    return statistics.median(ts)


n = 1024 * 1024  # 4MB
a = torch.randn(n, device="npu:0")
b = torch.randn(n, device="npu:0")
out = torch.empty_like(a)
grid = (triton.cdiv(n, 4096),)

t_native = host_time(lambda: torch.add(a, b))
t_triton = host_time(lambda: add_flat[grid](a, b, out, n, BLOCK=4096, num_warps=8))

import flag_gems
flag_gems.enable()
t_gems_same = host_time(lambda: torch.add(a, b))          # 快路径 (同形连续)
A2 = torch.randn(1024, 1, device="npu:0")
B2 = torch.randn(1, 1024, device="npu:0")
t_gems_bcast = host_time(lambda: torch.add(A2, B2))       # 慢路径 (广播)

print(f"[host enqueue 开销分解, 4MB, 中位数]")
print(f"  层1 原生 torch.add (C++):        {t_native:7.1f} us")
print(f"  层2 裸 triton launch (Python):   {t_triton:7.1f} us   <- triton 链路成本")
print(f"  层3 flag_gems add 同形(快路径):  {t_gems_same:7.1f} us   <- 框架层叠加成本 = {t_gems_same - t_triton:+.1f}")
print(f"  层3 flag_gems add 广播(慢路径):  {t_gems_bcast:7.1f} us")
print(f"\n  stage-2 总靶位 (层3-层1): {t_gems_same - t_native:.1f} us/次")
print(f"    其中 triton launch 链: {t_triton - t_native:.1f} us, flag_gems 框架层: {t_gems_same - t_triton:.1f} us")
