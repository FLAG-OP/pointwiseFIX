"""守卫副作用验证:
1. 共享 config 不被污染: 广播调用后 prefer_block_pointer 仍 True (bptr 环境)
2. bptr 正常路径仍走 bptr: 非广播调用的 kernel 缓存名含 _bptr
3. 单例身份不变: get_codegen_config() 返回同一对象
"""
import torch
import torch_npu
import flag_gems
from flag_gems.utils.codegen_config_utils import get_codegen_config

cfg = get_codegen_config()
cfg.prefer_1d_tile = False
cfg.prefer_block_pointer = True
orig_id = id(cfg)

flag_gems.enable()
A = torch.randn(128, 1, device="npu:0")
B = torch.randn(1, 128, device="npu:0")
M = torch.randn(128, 128, device="npu:0")

c = torch.add(A, B)  # 广播 -> 触发守卫
print("广播后 config.prefer_block_pointer =", cfg.prefer_block_pointer, "(应仍 True)")
print("config 单例身份不变 =", id(get_codegen_config()) == orig_id)

# 非广播仍应走 bptr (kernel 名含 _bptr)
import glob, os
before = set(glob.glob("/root/.flaggems/code_cache/*add_func_kernel_rank_2*"))
torch.add(M, M.clone())  # 非广播, bptr
torch.npu.synchronize()
after = set(glob.glob("/root/.flaggems/code_cache/*add_func_kernel_rank_2*"))
new = [os.path.basename(f) for f in after - before]
print("非广播新增 kernel:", [n[:70] for n in new])
print("非广播走 bptr =", any("_bptr_" in n for n in new), "(应 True)")

before = set(glob.glob("/root/.flaggems/code_cache/*add_func_kernel_rank_2*"))
torch.add(A, B)  # 广播 -> 线性寻址
torch.npu.synchronize()
after = set(glob.glob("/root/.flaggems/code_cache/*add_func_kernel_rank_2*"))
new = [os.path.basename(f) for f in after - before]
print("广播新增 kernel:", [n[:70] for n in new])
print("广播走线性 (无 bptr) =", all("_bptr_" not in n for n in new) if new else "复用缓存", "(应 True)")
