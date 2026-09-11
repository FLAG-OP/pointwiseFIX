"""stage-2 优化原型: prepare_args 快路径 memoization。

测量原型 (不改库): 手工模拟"缓存 prepare_args 结果"后,
跳过框架层重复计算能省多少。用直接持有 wrapper + 预构造 StridedBuffer 的
调用方式对照走完整 __call__ 的方式。
"""
import statistics
import time

import torch
import torch_npu
import flag_gems
from flag_gems.ops.add import add_func  # PointwiseDynamicFunction 实例

flag_gems.enable()
n = 1024 * 1024
a = torch.randn(n, device="npu:0")
b = torch.randn(n, device="npu:0")


def host_time(fn, iters=500):
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


# 完整路径: torch dispatch -> pointwise __call__ -> prepare_args -> wrapper
t_full = host_time(lambda: torch.add(a, b))

# 绕过 dispatch + prepare_args: 预构造参数, 直接调 overload (instantiate 一次)
ndim, args, kwargs, call_cfg = add_func.prepare_args(a, b, 1)
overload = add_func.instantiate(ndim)
t_direct = host_time(lambda: overload(*args, **kwargs))

# 更激进: 预先 unwrap —— overload 内部还会做 stride 读取等, 已无法再绕
print(f"[stage-2 原型: 4MB 同形 add, host enqueue]")
print(f"  完整路径 (dispatch+prepare_args+wrapper): {t_full:7.1f} us")
print(f"  直调 overload (省 dispatch+prepare_args): {t_direct:7.1f} us")
print(f"  可省: {t_full - t_direct:.1f} us/次 ({(t_full - t_direct)/t_full*100:.0f}%)")
# 注: 这是缓存的上限收益 (不含缓存 key 计算成本)
