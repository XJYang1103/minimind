import math
import torch
import torch.nn as nn

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))


# 2 条序列，每条 3 个 token，每个 token 8 维。
x=torch.randn(2,3,8)
bsz,seqlon,_=x.shape

# Q 有 4 个头，K/V 各有 2 个头，每头 2 维。
wq=nn.Linear(8,8,bias=False)
wk=nn.Linear(8,4,bias=False)
wv=nn.Linear(8,4,bias=False)
wo=nn.Linear(8,8,bias=False)
# 只在训练模式启动，调用 .eval()后会让p=0.0
attn_dropout = nn.Dropout(p=0.0) # p=0.1:训练时，每个注意力权重有 10% 的概率被置为 0。

# 1. 投影并拆头。
q=wq(x).reshape(bsz,seqlon,4,2)    #(2,3,4,2)
k=wk(x).reshape(bsz,seqlon,2,2)    #(2,3,2,2)
v=wv(x).reshape(bsz,seqlon,2,2)

# 2.归一化，旋转rope

# 3.GQA 每个 K/V 头重复两次，匹配 4 个 Q 头。
k=repeat_kv(k,2)
v=repeat_kv(v,2)

# 4.转置 或 q=torch.transpose(q,1,2)
q=q.transpose(1,2)  # 变成(2,4,3,2)  batch,head,token,dim
k=k.transpose(1,2)  # 变成(2,4,3,2)  
v=v.transpose(1,2)  # 变成(2,4,3,2)  因为此时kv已经repeat_kv了，所以也变成4个头了，跟q一样

# 5.计算得分
scores=(q@k.transpose(-2,-1)).float()/math.sqrt(k.shape[-1]) # 先用 FP32 处理分数和 softmax
print("scores:",scores.shape,"\n",scores)  

#6. 因果mask
mask=torch.full(
    (seqlon,seqlon),
    float("-inf"),           
    device=scores.device
).triu(1)

scores+=mask
print("scores:",scores.shape,"\n",scores)  
print("mask:\n",mask)

#7.softmax + dropout
weight=torch.softmax(scores,dim=-1).type_as(q)  # 跟上面的.float()对应，再转回跟q一样的
weight=attn_dropout(weight)
print("weight:\n",weight)

#8.加权求和
outpout1=weight @ v
print("output1的形状:",outpout1.shape)

#9.
#先把位置转置回来
outpout2=outpout1.transpose(1,2)
print("output2的形状:",outpout2.shape)
#拼接，融合映射
outpout3=outpout2.reshape(bsz,seqlon,-1)
outpout3=wo(outpout3)


print("一个头的权重：\n", weight[0, 0])
print("每行权重之和：", weight[0, 0].sum(-1))





