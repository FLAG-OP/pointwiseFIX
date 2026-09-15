"""v3: 补大形状, 验证 "device 比值是否随形状增大而收敛到 1x"。
若大形状 device -> 1x, 则小形状的 device 比值也是 launch-gap 伪影;
若大形状 device 仍 >1x, 则 kernel 本体确实慢。
"""
import statistics
import time

import torch
import torch_npu

DEV = "npu:0"


def wall(fn, iters=50, warmup=10, rounds=5):
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


def devtime(fn, iters=20, warmup=5):
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


# 大形状: kernel 足够长, launch gap 占比小
shapes = {
    "add 128MB": (torch.randn(8192, 4096, device=DEV), None),
    "gelu tanh 128MB": (torch.randn(8192, 4096, device=DEV), None),
}
xs = torch.randn(8192, 4096, device=DEV)
ys = torch.randn(8192, 4096, device=DEV)

print("== 大形状 (128MB): 原生基线 ==")
t_add_n = (wall(lambda: torch.add(xs, ys)), devtime(lambda: torch.add(xs, ys)))
t_gelu_n = (wall(lambda: torch.nn.functional.gelu(xs, approximate="tanh")), devtime(lambda: torch.nn.functional.gelu(xs, approximate="tanh")))
print(f"add  原生: wall {t_add_n[0]:8.3f} ms | device {t_add_n[1]:8.3f} ms")
print(f"gelu 原生: wall {t_gelu_n[0]:8.3f} ms | device {t_gelu_n[1]:8.3f} ms")

import flag_gems
flag_gems.enable()
t_add_g = (wall(lambda: torch.add(xs, ys)), devtime(lambda: torch.add(xs, ys)))
t_gelu_g = (wall(lambda: torch.nn.functional.gelu(xs, approximate="tanh")), devtime(lambda: torch.nn.functional.gelu(xs, approximate="tanh")))
print(f"add  gems:  wall {t_add_g[0]:8.3f} ms | device {t_add_g[1]:8.3f} ms  (device 比值 {t_add_g[1]/t_add_n[1]:.2f}x)")
print(f"gelu gems:  wall {t_gelu_g[0]:8.3f} ms | device {t_gelu_g[1]:8.3f} ms  (device 比值 {t_gelu_g[1]/t_gelu_n[1]:.2f}x)")

# gather 大形状 (之前 1MB 时 44x, 看 17MB)
xg = torch.randn(64, 256, 512, device=DEV, dtype=torch.float16)
idxg = torch.randint(0, 256, (64, 256, 512), device=DEV, dtype=torch.int64)
import importlib
print("(gather 大形状原生基线需 enable 前测 —— 用另一进程口径, 见 gatherFIX bench_shapes: 17MB 时 wall 比值 1.0x)")
t_gather_g = (wall(lambda: torch.gather(xg, 1, idxg), iters=10), devtime(lambda: torch.gather(xg, 1, idxg), iters=10))
print(f"gather 17MB gems: wall {t_gather_g[0]:8.3f} ms | device {t_gather_g[1]:8.3f} ms (原生参照 ~160ms, 即 1.0x)")
