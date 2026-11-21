import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSwiGLU_FFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

ffn = SimpleSwiGLU_FFN(768, 2048)
x = torch.randn(2, 10, 768)
output = ffn(x)
print(f"输入形状: {x.shape}")        # (2, 10, 768)
print(f"输出形状: {output.shape}")    # (2, 10, 768)
print(f"参数量: {sum(p.numel() for p in ffn.parameters()):,}")
