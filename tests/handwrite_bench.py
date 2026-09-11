"""手写调优 Triton add vs flag_gems codegen vs 原生 torch_npu。

调优思路 (针对 Ascend 910, 48 AI core, bptr 关闭):
  - 1D 展平 + 大 BLOCK (4096/8192) —— 内存带宽型算子要点是喂满搬运
  - num_warps 高值 (向量核多用并行束)
  - ILP: 每个 program 处理多段 (ELEMS_PER_CTA), 减少启动/调度开销
"""
import statistics
import time

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def add_flat_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr, EPC: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK * EPC
    for i in tl.static_range(EPC):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m)
        y = tl.load(y_ptr + offs, mask=m)
        tl.store(o_ptr + offs, x + y, mask=m)


def tri_add(a, b, BLOCK=8192, EPC=4, warps=8):
    out = torch.empty_like(a)
    n = a.numel()
    grid = (triton.cdiv(n, BLOCK * EPC),)
    add_flat_kernel[grid](a, b, out, n, BLOCK=BLOCK, EPC=EPC, num_warps=warps)
    return out


def bench(fn, iters=100, warmup=30, rounds=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(ts)


for desc, sa, sb in [
    ("(4096,4096) fp32 同形 128MB", (4096, 4096), (4096, 4096)),
    ("(2048,2048) fp32 同形 32MB", (2048, 2048), (2048, 2048)),
]:
    a = torch.randn(*sa, device="npu:0")
    b = torch.randn(*sb, device="npu:0")
    ref = a + b
    got = tri_add(a, b)
    ok = torch.equal(ref, got)
    t_native = bench(lambda: torch.add(a, b))
    t_tri = bench(lambda: tri_add(a, b))
    bw = (a.numel() * 4 * 3) / (t_tri / 1e6) / 1e9  # GB/s: r+w 3x
    print(f"[{desc}] 数值正确={ok}")
    print(f"  原生 torch_npu:      {t_native:8.1f} us  ({a.numel()*4*3/(t_native/1e6)/1e9:6.0f} GB/s)")
    print(f"  手写 Triton (flat):  {t_tri:8.1f} us  ({bw:6.0f} GB/s)  比值 {t_tri/t_native:.2f}x")

# 广播场景: (4096,1)+(1,4096)
a = torch.randn(4096, 1, device="npu:0")
b = torch.randn(1, 4096, device="npu:0")
t_native = bench(lambda: torch.add(a, b))
print(f"[广播 (4096,1)+(1,4096)]")
print(f"  原生 torch_npu: {t_native:8.1f} us")
