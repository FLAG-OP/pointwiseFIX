"""issue #5739 复现: 广播 add (128,1)+(1,128) -> 'strides must not be zero'。"""
import torch
import torch_npu
import flag_gems

a = torch.randn(128, 1, device="npu:0")
b = torch.randn(1, 128, device="npu:0")
c1 = torch.add(a, b)  # enable 前: 原生 OK
print("native add ok:", tuple(c1.shape))

flag_gems.enable()
try:
    c2 = torch.add(a, b)
    torch.npu.synchronize()
    ok = torch.allclose(c1.cpu(), c2.cpu())
    print("gems add ok:", tuple(c2.shape), "allclose:", ok)
except Exception as e:
    print(f"GEMS ADD FAILED: {type(e).__name__}: {str(e)[:200]}")
