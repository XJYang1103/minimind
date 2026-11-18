import torch

def my_rmsnorm(x,weight,eps=1e-6):
    rms=torch.sqrt(x.pow(2).mean(-1,keepdim=True)+eps)
    return weight*x/rms

x = torch.randn(2, 2, 4)
weight = torch.ones(4)
output = my_rmsnorm(x, weight)
print(output.shape)  

print(x)
print(output)

print(x.pow(2).mean(-1))
#在 γ 全为 1、ε 影响很小时，每个 token 的所有特征“平方后求平均”会接近 1
print(output.pow(2).mean(-1)) 





print()
import torch.nn as nn

x = torch.randn(2, 4, 768)
ln = nn.LayerNorm(768)
# RMSNorm 需要自定义（上面的类）

ln_out = ln(x)
rms_out = my_rmsnorm(x, torch.ones(768))

# 两者输出会不同，但统计特性类似
print(f"LayerNorm 均值: {ln_out.mean(-1).mean():.4f}")   # ≈ 0
print(f"RMSNorm 均值: {rms_out.mean(-1).mean():.4f}")     # ≠ 0（没有减均值）

