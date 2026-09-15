"""原生 vs flag_gems 全景性能对照 v2 (五库核心算子统一矩阵).

口径: wall (perf_counter+每轮sync, 含host链) 和 device (NPU events, 纯kernel) 双报。
顺序: 每个算子块内, 原生在 enable 前测, gems 在 enable 后测 (同进程先后)。
为避免 enable 状态切换问题: 全部原生基线先测完, 再 enable 测全部 gems 侧。
"""
import statistics
import time

import torch
import torch_npu

DEV = "npu:0"


def wall(fn, iters=100, warmup=30, rounds=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(ts)


def devtime(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) / iters


CASES = [
    ("gelu tanh 4MB",        lambda x4m, a, b, xg, idxg, xo, xe: torch.nn.functional.gelu(x4m, approximate="tanh")),
    ("add 同形 4MB",         lambda x4m, a, b, xg, idxg, xo, xe: torch.add(x4m, x4m)),
    ("add 广播 (128,1)+(1,128)", lambda x4m, a, b, xg, idxg, xo, xe: torch.add(a, b)),
    ("gather 连续 1MB fp16", lambda x4m, a, b, xg, idxg, xo, xe: torch.gather(xg, 1, idxg)),
    ("one_hot (4k,)x1000类", lambda x4m, a, b, xg, idxg, xo, xe: torch.nn.functional.one_hot(xo, 1000)),
    ("lift_fresh 非空 4MB",  lambda x4m, a, b, xg, idxg, xo, xe: torch.ops.aten.lift_fresh(x4m)),
]

x4m = torch.randn(1024, 1024, device=DEV)
a = torch.randn(128, 1, device=DEV)
b = torch.randn(1, 128, device=DEV)
xg = torch.randn(16, 64, 256, device=DEV, dtype=torch.float16)
idxg = torch.randint(0, 64, (16, 64, 256), device=DEV, dtype=torch.int64)
xo = torch.randint(0, 1000, (4096,), device=DEV, dtype=torch.int64)
xe = torch.empty(0, device=DEV)
args = (x4m, a, b, xg, idxg, xo, xe)

print("=" * 76)
print("原生 vs flag_gems 全景对照 [wall=含host链 | device=纯kernel (NPU events)]")
print("=" * 76)

# ---- 全部原生基线 (enable 前) ----
native = {}
for name, fn in CASES:
    native[name] = (wall(lambda: fn(*args)), devtime(lambda: fn(*args)))

# 空张量 (只有 wall 有意义)
t_ne = wall(lambda: torch.ops.aten.lift_fresh(xe), iters=300)

import flag_gems
flag_gems.enable()

# ---- gems 侧 ----
for name, fn in CASES:
    tw_n, td_n = native[name]
    tw_g, td_g = wall(lambda: fn(*args)), devtime(lambda: fn(*args))
    print(f"[{name}]")
    print(f"  原生: wall {tw_n:8.4f} ms | device {td_n:8.4f} ms")
    print(f"  gems:  wall {tw_g:8.4f} ms | device {td_g:8.4f} ms")
    print(f"  比值:  wall {tw_g/tw_n:6.1f}x | device {td_g/td_n:6.1f}x")

t_ge = wall(lambda: torch.ops.aten.lift_fresh(xe), iters=300)
print(f"[lift_fresh 空张量]")
print(f"  原生: wall {t_ne*1e3:6.1f} us | gems: wall {t_ge*1e3:6.1f} us ({t_ge/t_ne:.1f}x; 修复前=进程崩)")
print("\n注: wall 与 device 比值的分离 = host 链路开销占比 (pointwise 类 ~9-10x wall / ~1x device 预期)")
