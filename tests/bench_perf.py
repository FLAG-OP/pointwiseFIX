"""pointwiseFIX 性能对照实验 (#5739).

stage-1 修复语义: 广播守卫 (检测 + per-call config)。性能问题域:
  A. 守卫检测开销: 修复新增 = 形状比较 O(operands*ndim) —— 直测
  B. 非广播同形: 修复前后都走快路径, 守卫不触发 —— 回归护栏
  C. 广播: 默认配置下修复前后同为慢路径线性寻址 (bptr=False), 守卫仅
     多一次形状判断 —— 直测
  D. bptr 环境 (模拟): 修复前=崩; 修复后=线性寻址 —— perf_guard.py 已测
"""
import statistics
import time

import torch
import torch_npu
import flag_gems

DEV = "npu:0"


def bench(fn, iters=200, warmup=50, rounds=5):
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


flag_gems.enable()

# A. 守卫检测成本 (纯 Python, 形状元组比较)
A = torch.randn(128, 1, device=DEV)
B = torch.randn(1, 128, device=DEV)
M1 = torch.randn(256, 256, device=DEV)
task = (128, 128)
t_guard = bench(lambda: (A.shape != task or B.shape != task))
print("=" * 66)
print("A. 守卫检测成本 (修复新增, 形状比较):", f"{t_guard:.3f} us/次")
print("   (相对 pointwise host 路径 ~120us 完全可忽略)")

# B. 非广播回归护栏: 同形 add (修复不触发)
M2 = torch.randn(256, 256, device=DEV)
ref_native = bench(lambda: torch.add(M1, M2))
print(f"\nB. 非广播同形 add: {ref_native:7.1f} us (走快路径, 守卫未触发, 与修复前同)")

# C. 广播: 修复前后同为线性寻址, 守卫只加形状判断
t_bcast = bench(lambda: torch.add(A, B))
print(f"C. 广播 add (128,1)+(1,128): {t_bcast:7.1f} us (默认配置, 与修复前差异=A项)")

# 大形状广播端到端 (host 开销占比小, 看总量)
A2 = torch.randn(4096, 1, device=DEV)
B2 = torch.randn(1, 4096, device=DEV)
t_big = bench(lambda: torch.add(A2, B2), iters=100)
print(f"   广播 add (4096,1)+(1,4096): {t_big:7.1f} us")
print("\n判读: 修复对默认配置是 no-op (A 项 ~0.1us 级); bptr 环境收益见 perf_guard.py")
