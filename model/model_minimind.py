import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5): #dim：特征维度，MiniMind 中是 768
        super().__init__()
        self.eps = eps  # epsilon防止除0异常
        self.weight = nn.Parameter(torch.ones(dim)) # γ(768,)，初始化全1

    def norm(self, x): #计算 没乘γ前的
        # mean(-1, )是沿着最后一维求平均，保留前面的batch，token等维度
        # keepdim=True 是保留求平均的这个维度，把他的大小变成1：
        """
        输入 x                     (2, 3, 768)
        mean(-1, keepdim=True)     (2, 3, 1)  这样让最后一维是1，方便和x进行广播
        mean(-1, keepdim=False)    (2, 3)
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x): # 乘γ。
        """
        .type_as(x): 把结果转变成x的输入类型，传回x

        原始 x       float16，形状 (2, 3, 768)
        x.float()      float32，形状 (2, 3, 768)
        归一化计算      float32，形状 (2, 3, 768)
        .type_as(x)  float16，形状 (2, 3, 768)

        归一化先用 FP32 计算，再将结果转回原输入 x 的类型
        """
        return (self.weight * self.norm(x.float())).type_as(x)


# RoPE 实现的核心函数，预计算所有位置的旋转角度cos、sin 表（跟minimind-learn的不一样）
# 这里是 前后半维配对，而learn是 相邻维度配对
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    #频率 θ ,此时是针对所有维度对i的，形状是（dim/2,）
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    # YaRN
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    # t就是p的序列，把p和θ 逐个相乘得到最终的角度
    t = torch.arange(end, device=freqs.device)
    # p·θ得到角度α
    freqs = torch.outer(t, freqs).float()
    # cat用来沿着dim=-1拼接向量，attn_factor 是YaRN缩放系数。
    # 算两遍一样的cos，拼接到一起
    """
    freqs=每个维度对i处的 角度，记录所有token（p），所有维度对（i），的角度p·θ_i

    假设有3个token，两个维度对：
    freqs =
    [
        [0, 0],      p=0
        [1, 0.001],  p=1
        [2, 0.002]   p=2
    ]

    cat([torch.cos(freqs), torch.cos(freqs)], dim=-1):
    某个位置的角度       [α, β]
    求 sin              [sinα, sinβ]
    拼接两份            [sinα, sinβ, sinα, sinβ]


    为什么要拼接两份？
    因为这份实现将前后两半配对。（learn里面是***相邻配对***，就是第一个和第二个配对）
    特征       [a,    b,    c,    d]
    配对        (a,c) 使用角度 α
                (b,d) 使用角度 β

    cos 系数   [cosα, cosβ, cosα, cosβ]
    sin 系数   [sinα, sinβ, sinα, sinβ]
    这样 a、c 得到相同的角度系数，b、d 也得到相同的角度系数，就能配合 rotate_half 完成二维旋转。
    
    """
    # 得到角度的sin，cos值
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
# 假设q用一个batch二条,每条2个 token、每个token 2个头、每头8维。q.shape=(2, 2, 2, 8)
    def rotate_half(x):  # 假设只用最后一维举例 x = [1, 2, 3, 4, 5, 6, 7, 8]
        return torch.cat(
            (   # ...是保留保留前面的所有维度，只对最后一维切割  
                -x[..., x.shape[-1] // 2:], #8//2=4，从下标为4处开始直到最后：[−5, −6, −7, −8]
                x[..., : x.shape[-1] // 2]  #左闭右开，取不到下标4：[1, 2, 3, 4]
            ), 
            dim=-1) # [−5, −6, −7, −8, 1, 2, 3, 4]
    # 对k，q执行旋转
    """
    假设 q = [1, 2, 3, 4, 5, 6, 7, 8]
    前后半维配成 4 对。
    第0对：(1, 5)
    第1对：(2, 6)
    第2对：(3, 7)
    第3对：(4, 8)  对每一对(1x2)的向量进行旋转

    上一个函数已经得到了sin，cos对应角度的值:假设四个角度是90°、0°、90°、0°。
    cos = [0, 1, 0, 1, 0, 1, 0, 1]
    sin = [1, 0, 1, 0, 1, 0, 1, 0]  

1.
    x     [1, 2, 3, 4, 5, 6, 7, 8]
    cos   [0, 1, 0, 1, 0, 1, 0, 1]
    结果  [*0, 2, 0, 4, *0, 6, 0, 8]
2.
    rotate_half(x)  [−5, −6, −7, −8, 1, 2, 3, 4]
    sin            [ 1,  0,  1,  0, 1, 0, 1, 0]
    结果           [*−5,  0, −7,  0, *1, 0, 3, 0]
3.
    x * cos                 [ 0, 2,  0, 4, 0, 6, 0, 8]
    rotate_half(x) * sin    [−5, 0, −7, 0, 1, 0, 3, 0]
                            ─────────────────────────
    最终结果                 [*−5, 2, −7, 4, *1, 6, 3, 8]  (*是第0对的值)

| 原来的二维对 | 角度 | 旋转后 |
| `(1,5)` | 90° | `(−5,1)` |
| `(2,6)` | 0°  | `(2,6)` |
| `(3,7)` | 90° | `(−7,3)` |
| `(4,8)` | 0°  | `(4,8)` |

其中：
    R(a)=   [    
                [cosa , -sina ]  
                [sina , cosa  ]
            ]

    对于第0对q=(1, 5)来说：
    q'=R(a)q=   [    
                    [cosa , -sina ]  ·  [0,5]T =   [-5,1]T
                    [sina , cosa  ]
                ]
    因为只有 -sina是需要负数的， -sina只对5相乘，所以只需要对后半部分加符负号
            
    """
# unsqueeze在指定位置插入一个大小为1的维度，用于广播
    # cos：
    # 原来    (3,    96)
    # 之后    (3, 1, 96)    unsqueeze_dim这里是1，所以在cos的1处插入一个大小为1的维度
# cos需要和q想乘，需要广播，让96维能广播到每一个头上：
    # Q                 (2, 3, 8, 96)
    # cos               (1, 3, 1, 96)  第一个1是广播的，第二个1是插入的
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))


# MiniMindModel--(layer)-->MiniMindBlock--(config)-->Attention
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True

        #权重矩阵 Wq,Wk,Wv,Wo。开始时随机初始化self.q_proj.weight
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape #若x.shape = (2, 3, 768)，则取出这三个值
        #1. proj映射，用权重矩阵映射 x，变成QKV矩阵
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        #2. 划分多头
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)     #..8*96
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  #..4*96
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  #..4*96

        #3. 对 Q K做归一化，pre-norm
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        # 取出从MiniMindModel一路传递过来的 cos，sin 表
        cos, sin = position_embeddings
        #4. ROPE旋转 Q K
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # KV Cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None


        #5. 先把kv拓展头数(4->8)，跟q的头数匹配。然后再转置1,2维：
        # 转置前  (2, 3, 8, 96)  batch、token、--头、特征
        # 转置后  (2, 8, 3, 96)  batch、头、--- token、特征
        # 让每个头内部单独计算 所有token的q，找所有token的k
        xq, xk, xv = (  xq.transpose(1, 2), 
                        repeat_kv(xk, self.n_rep).transpose(1, 2), 
                        repeat_kv(xv, self.n_rep).transpose(1, 2))

        # 自动完成attention计算
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)

        else: #6. 注意力分数 Q=(2, 8, 3, 96)，Kᵀ=(2, 8, 96, 3) ，所以scores的行是q，列是k
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            # 7.因果遮罩，屏蔽未来位置
            if self.is_causal: 
                scores[:, :, :, -seq_len:] += (
                torch.full(                  # torch.full 创建并填满矩阵
                    (seq_len, seq_len),      #矩阵形状 3*3
                    float("-inf"),           # 用什么填充
                    device=scores.device
                    )
                    .triu(1)  
                # triu 表示取上三角，参数 1 表示从主对角线上方第一条对角线开始保留。其他位置变成 0。 
                # [0, −∞, −∞]
                # [0,  0, −∞]
                # [0,  0,  0]
                )
            # Padding 遮罩，屏蔽无效输入
            if attention_mask is not None: 
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            #1.softmax变成概率分布，scores.shape(2, 8, 3, 3)，8个qkv头(已拓展)，3是token数
            """
            dim=-1是对列（key）操作，每一行的三个数和为1
            scores的最后两维：
                                Key 0   Key 1   Key 2
                        q0       1      −∞      −∞
                        q1       1       2      −∞
                        q2       1       2       3
            softmax:
                    [1.0000, 0.0000, 0.0000],   和为1
                    [0.2689, 0.7311, 0.0000],   和为1
                    [0.0900, 0.2447, 0.6652]    和为1
            """
            #2.dropout，在训练时随机丢弃部分权重
            #3.计算attention：(..3*3)跟v(3*96)加权求和,得到 output.shape = (2, 8, 3, 96)
            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq))  @ xv
        #把头和token的维度换回来， (2, 8, 3, 96) → (2, 3, 8, 96)
        # -1：拼接8个头分别计算出来的注意力向量    (2, 3, 8, 96) → (2, 3, 768)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 拼接之后的注意力，再经过一下映射(768*768)融合起来，然后经过一次dropout
        output = self.resid_dropout(self.o_proj(output)) #这里还没有执行残差相加。相加发生在外面的 MiniMindBlock 中。
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size # 升维后的度数
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1: 
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]
                topk_weight = top1 - top1.detach() + 1.0 # k=1: 1.0 forward on purpose, gradient via straight-through
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

# 送进 self.layers里面作为一层
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config) # 把block从model.layer接收的值传到atten

        # 归一化，不改变原有的维度数，只是让数值更平稳
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 按配置选择普通 FFN 或 MoE
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states # 保持原始输入x，用于残差连接

        # x → Norm → Attention → Add(x)
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, #先归一化再atten
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        # Norm → FFN → Add(x)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        #embed_tokens
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 初始化模型时，提前生成完整cos，sin表
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # embedding嵌入，把ID转化成向量：input_ids由(1,3)--->(1,3,768) 假设batchsize=1条序列，3个token
        # dropout：在训练时，随机把向量中的部分元素置为 0，剩余元素乘上放大因子，使输出的期望值与输入一致。
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        #前向计算时，截取当前的位置，送入block，block再送入atten
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []

        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        # 经过8个layer(block)之后，最后一个block做了Norm → FFN → Add(x)，此时还需要一个归一化调整add之后的
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

# 用来真正实现训练的类，包括预训练和SFT，完成预测下一个词的任务
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        #lm_head 将每个位置的隐藏向量转换为词表分数（最后一层的线性层，将向量变回词）
        ## lm_head：隐藏向量 → 词表分数 logits，不是直接得到 token
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        if self.config.tie_word_embeddings: 
            # embed，lm_head共享权重
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        # 经过model，得到最终的隐藏状态
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # 最后一个线性层，接收atten分数，得到词表得分：(2, 6, 768) → (2, 6, 6400) 词表大小6400
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None

        if labels is not None:
            # x是全部的分数，但是labels要向后错开一个token，保证labels是要预测的值
            x = logits[..., :-1, :].contiguous() # (2, 5, 6400)  batch、预测位置、词表分数
            y = labels[..., 1:].contiguous()   # (2, 5)   每个位置的目标 ID
            loss = F.cross_entropy(
                # 展平xy，view和reshape都需要一个新的对象接收返回值，不改变原来xy的形状
                x.view(-1, x.size(-1)),  #(2, 5, 6400) → (10, 6400)，x.size(-1)是最后一维的大小
                y.view(-1),              #(2, 5)       → (10,)
                ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    # 执行多次 forward，不断的预测下一个词
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids