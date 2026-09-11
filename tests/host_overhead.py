"""量化 Triton host 侧开销分解: 缓存命中后每次调用到底慢在哪。

阶段:
  1. kernel 热缓存纯 launch (同参重复调) —— launch 链路开销
  2. 换 out 张量 (arg 变化, 仍缓存命中) —— arg 打包开销
  3. flag_gems codegen wrapper 层额外开销 (对比裸 jit)
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


n = 1024 * 1024  # 4MB: kernel 本体 ~15-300us, host 开销占比可见
a = torch.randn(n, device="npu:0")
b = torch.randn(n, device="npu:0")
out = torch.empty_like(a)
grid = (triton.cdiv(n, 4096),)


def bench(fn, iters=300, warmup=50):
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


# 1. 热缓存, 全同参
t_same = bench(lambda: add_flat[grid](a, b, out, n, BLOCK=4096, num_warps=8))

# 2. 每次 new out (arg 变)
t_newout = bench(lambda: add_flat[grid](a, b, torch.empty_like(a), n, BLOCK=4096, num_warps=8))

# 3. 原生同尺寸
t_native = bench(lambda: torch.add(a, b))

# 4. 纯 empty_like 分配开销
t_alloc = bench(lambda: torch.empty_like(a))

# 5. host-only 计时: launch 是异步的, 前后同步只做一次 —— 测纯 host 排队时间
def host_only():
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(300):
        add_flat[grid](a, b, out, n, BLOCK=4096, num_warps=8)
    t1 = time.perf_counter()
    torch.npu.synchronize()
    return (t1 - t0) / 300 * 1e6

t_host = host_only()

print(f"[4MB, 热缓存]")
print(f"  原生 torch.add:            {t_native:8.1f} us")
print(f"  triton 同参重调 (端到端):  {t_same:8.1f} us")
print(f"  triton 每次新 out:         {t_newout:8.1f} us  (含分配 {t_alloc:.1f})")
print(f"  triton 纯 host 排队时间:   {t_host:8.1f} us  <- Python launch 链路开销")
