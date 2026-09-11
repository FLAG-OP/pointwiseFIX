"""扩大复现矩阵: 广播 pointwise 的各种形状组合, 找 'strides must not be zero' 触发条件。

issue 原例 (128,1)+(1,128) 在本机不触发。矩阵覆盖:
  - 不同 rank (1-5)
  - 广播维度在不同位置
  - 不同算子 (add/sub/le/gt/mul)
  - 不同 tile 策略触发不同 kernel 实例化路径 (rank 变化会生成新 kernel)
"""
import torch
import torch_npu
import flag_gems

flag_gems.enable()

fails = []
total = 0

def t(name, fn):
    global total
    total += 1
    try:
        r = fn()
        torch.npu.synchronize()
        return r
    except Exception as e:
        msg = str(e)
        fails.append((name, f"{type(e).__name__}: {msg[:80]}"))
        print(f"  [FAIL] {name}: {type(e).__name__}: {msg[:100]}")
        return None

cases = [
    ("issue原例 (128,1)+(1,128)", lambda: torch.add(torch.randn(128, 1, device="npu:0"), torch.randn(1, 128, device="npu:0"))),
    ("(3,1)+(1,4) 小形状", lambda: torch.add(torch.randn(3, 1, device="npu:0"), torch.randn(1, 4, device="npu:0"))),
    ("(1,)+(128,) 右广播", lambda: torch.add(torch.randn(1, device="npu:0"), torch.randn(128, device="npu:0"))),
    ("(128,)+(1,) 左广播", lambda: torch.add(torch.randn(128, device="npu:0"), torch.randn(1, device="npu:0"))),
    ("(128,1,1)+(1,64,32) 3d", lambda: torch.add(torch.randn(128, 1, 1, device="npu:0"), torch.randn(1, 64, 32, device="npu:0"))),
    ("(1,64,32)+(128,1,1) 3d 反", lambda: torch.add(torch.randn(1, 64, 32, device="npu:0"), torch.randn(128, 1, 1, device="npu:0"))),
    ("标量广播 (128,128)+0d", lambda: torch.add(torch.randn(128, 128, device="npu:0"), torch.tensor(1.5, device="npu:0"))),
    ("expand 视图后加", lambda: torch.add(torch.randn(128, 1, device="npu:0").expand(128, 128), torch.randn(128, 128, device="npu:0"))),
    ("sub 广播", lambda: torch.sub(torch.randn(128, 1, device="npu:0"), torch.randn(1, 128, device="npu:0"))),
    ("le 广播", lambda: torch.le(torch.randn(128, 1, device="npu:0"), torch.randn(1, 128, device="npu:0"))),
    ("gt 广播", lambda: torch.gt(torch.randn(128, 1, device="npu:0"), torch.randn(1, 128, device="npu:0"))),
    ("mul 广播", lambda: torch.mul(torch.randn(128, 1, device="npu:0"), torch.randn(1, 128, device="npu:0"))),
    ("混合 rank (2d+1d)", lambda: torch.add(torch.randn(128, 128, device="npu:0"), torch.randn(128, device="npu:0"))),
    # 注: 空张量用例移除 —— randn(0) 踩已知未修的零 grid bug (randn.py:145), 与本 issue 无关
    ("5d 广播", lambda: torch.add(torch.randn(2, 1, 1, 1, 1, device="npu:0"), torch.randn(1, 3, 4, 5, 6, device="npu:0"))),
    ("大张量广播 (2049,1)+(1,2049)", lambda: torch.add(torch.randn(2049, 1, device="npu:0"), torch.randn(1, 2049, device="npu:0"))),
]

for name, fn in cases:
    t(name, fn)

print(f"\nTOTAL: {total}, failed: {len(fails)}")
if not fails:
    print("全部通过: 本机环境未复现 issue (torch 2.10 vs issue 的 2.9)")
