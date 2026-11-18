import torch
import torch.nn as nn

# 5 个 token，每个 token 对应一个 3 维向量。
embedding = nn.Embedding(5, 3)

# 一个 batch，两条序列，每条三个 token。
input_ids = torch.tensor([
    [1, 3, 1],
    [0, 2, 4],
])

vectors=embedding(input_ids)

print(vectors,"\n")
print(embedding,"\n")
print(vectors.shape,"\n")

print("权重表形状：", embedding.weight.shape)
print("输入形状：", input_ids.shape)
print("输出形状：", vectors.shape)

# 对比 ID 1 查出的向量和表中第 1 行。
print("查表结果：", vectors[0, 0])
print("权重表第 1 行：", embedding.weight[1])