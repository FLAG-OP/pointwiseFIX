"""TDD 红: 广播 pointwise × bptr 强制开启 的回归矩阵 (修复前预期 FAIL)。

覆盖: 广播位置(前/后/中间/全) × 算子(add/sub/mul/le/gt) × 强制 bptr。
修复后应全绿: 广播时该次调用自动禁用 bptr 走线性寻址。
"""
import sys

import torch
import torch_npu
import flag_gems
from flag_gems.utils.codegen_config_utils import get_codegen_config

# 模拟 bptr 被激活的环境 (NVIDIA 默认 / 老 triton-ascend / 用户显式开启)
cfg = get_codegen_config()
cfg.prefer_1d_tile = False
cfg.prefer_block_pointer = True

flag_gems.enable()
results = []


def check(name, fn):
    try:
        r = fn()
        torch.npu.synchronize()
        results.append((name, True))
        print(f"  [PASS] {name} shape={tuple(r.shape)}")
    except Exception as e:
        results.append((name, False))
        print(f"  [FAIL] {name}: {type(e).__name__}: {str(e)[:70]}")


A = torch.randn(128, 1, device="npu:0")
B = torch.randn(1, 128, device="npu:0")
M = torch.randn(128, 128, device="npu:0")

# 1. issue 原例 + 各广播位置
check("add (128,1)+(1,128) [issue原例]", lambda: torch.add(A, B))
check("add (1,128)+(128,1) 交换", lambda: torch.add(B, A))
check("add (128,1)+同形M 尾维广播", lambda: torch.add(A, M))
check("add 同形M+(128,1)", lambda: torch.add(M, A))
check("add (128,1,1)+(1,64,32) 3d", lambda: torch.add(torch.randn(128, 1, 1, device="npu:0"), torch.randn(1, 64, 32, device="npu:0")))
check("add 标量0d广播", lambda: torch.add(M, torch.tensor(1.5, device="npu:0")))

# 2. 其他 pointwise 算子 (issue 提到的)
check("sub 广播", lambda: torch.sub(A, B))
check("mul 广播", lambda: torch.mul(A, B))
check("le 广播", lambda: torch.le(A, B))
check("gt 广播", lambda: torch.gt(A, B))

# 3. 非广播对照 (bptr 下应保持 OK, 证明修复不影响 bptr 正常路径)
check("add 同形 (bptr 正常路径)", lambda: torch.add(M, M.clone()))

# 4. 数值正确性 (修复后广播也要算对)
def add_val():
    ref = A.cpu() + B.cpu()
    got = torch.add(A, B).cpu()
    return got if torch.allclose(ref, got, atol=1e-5) else (_ for _ in ()).throw(AssertionError("数值错"))
check("广播数值正确", add_val)

n_fail = sum(1 for _, c in results if not c)
print(f"\nTOTAL: {len(results)}, failed: {n_fail}")
sys.exit(1 if n_fail else 0)
