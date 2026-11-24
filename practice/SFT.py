# trainer\train_full_sft.py
# dataset\lm_dataset.py
# model\model_minimind.py的最后的forward

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)

# ==================================================
# 1. 对话 → 模板文本 → Token ID
# ==================================================

# 两条人工构造的训练样本
samples = [
    [
        {"role": "user", "content": "你 好"},
        {"role": "assistant", "content": "你 好"},
    ],
    [
        {"role": "user", "content": "一 加 一"},
        {"role": "assistant", "content": "二"},
    ],
]


def apply_chat_template(messages):
    """教学模板：角色标记 + 内容 + 结束标记。"""
    parts = []
    for message in messages:
        parts.append(f"<{message['role']}>")
        parts.append(message["content"])
        parts.append("<end>")
    return " ".join(parts)


texts = [apply_chat_template(messages) for messages in samples]

# 教学 Tokenizer：按空格切分，再查词表
tokens = sorted(set(" ".join(texts).split()))
vocab = ["<pad>"] + tokens

token_to_id = {token: i for i, token in enumerate(vocab)}
id_to_token = {i: token for token, i in token_to_id.items()}

pad_id = token_to_id["<pad>"]
assistant_id = token_to_id["<assistant>"]
end_id = token_to_id["<end>"]
vocab_size = len(vocab)


def encode(text):
    return [token_to_id[token] for token in text.split()]


# ==================================================
# 2. Dataset：构造 input_ids 和 labels
# ==================================================

class ToySFTDataset(Dataset):
    def __init__(self, samples, max_length=12):
        self.samples = samples
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 消息列表 → 模板字符串 → ID 列表
        text = apply_chat_template(self.samples[index])
        input_ids = encode(text)[:self.max_length]

        # 右侧补 PAD
        input_ids += [pad_id] * (self.max_length - len(input_ids))

        # 默认所有位置都不参与 Loss
        labels = [-100] * len(input_ids)
        in_answer = False

        for i, token_id in enumerate(input_ids):
            if token_id == assistant_id:
                in_answer = True
                continue  # assistant 开头标记本身不作为监督目标

            if in_answer:
                if token_id == pad_id:
                    break

                labels[i] = token_id  # 保留回答正文和结束标记

                if token_id == end_id:
                    in_answer = False

        # 单条样本形状均为 (12,)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


dataset = ToySFTDataset(samples)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

# 检查第一条样本
ids, labels = dataset[0]

print("模板文本：", texts[0])
print("input_ids：", ids.tolist())
print("labels：   ", labels.tolist())

print("\n检查下一个 token 的监督关系：")
for i in range(len(ids) - 1):
    current_token = id_to_token[ids[i].item()]
    next_token = id_to_token[ids[i + 1].item()]
    target = labels[i + 1].item()
    print(
        f"{current_token:12s} → {next_token:12s}"
        f" label={target:4d}  参与Loss={target != -100}"
    )


# ==================================================
# 3. 微型因果语言模型：输入 → logits → Loss
# ==================================================

class TinyLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=16, max_length=12):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)

        # 单头 Attention，便于串起已经学过的计算
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device)

        # (B,T) → (B,T,16)
        h = self.embedding(input_ids) + self.position_embedding(positions)

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        # (B,T,16) @ (B,16,T) → (B,T,T)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])

        # 不允许看到未来
        causal_mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=h.device
        ).triu(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # 不允许关注作为 Key 的 PAD
        padding_mask = input_ids.eq(pad_id).unsqueeze(1)  # (B,1,T)
        scores = scores.masked_fill(padding_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        attention_output = self.o_proj(weights @ v)

        h = self.norm(h + attention_output)

        # (B,T,16) → (B,T,V)，得到词表分数
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            # 当前输出预测下一个 token
            predictions = logits[:, :-1, :]  # (B,T-1,V)
            targets = labels[:, 1:]           # (B,T-1)

            # 展平，保持预测与目标一一对应
            loss = F.cross_entropy(
                predictions.reshape(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss


# ==================================================
# 4. 训练：Loss → 梯度 → 参数更新
# ==================================================

model = TinyLM(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# 保存一份参数，后面检查是否真的发生更新
weight_before = model.lm_head.weight.detach().clone()

model.train()
optimizer.zero_grad(set_to_none=True)

for epoch in range(20):
    for input_ids, labels in loader:
        logits, loss = model(input_ids, labels=labels)

        if epoch == 0:
            print("\nbatch 输入形状：", input_ids.shape)  # (2,12)
            print("batch 标签形状：", labels.shape)       # (2,12)
            print("logits 形状：", logits.shape)          # (2,12,V)
            print("Loss 形状：", loss.shape)              # 标量：()

        loss.backward()  # 计算梯度，尚未更新参数

        if epoch == 0:
            print(
                "LM Head 梯度范数：",
                model.lm_head.weight.grad.norm().item()
            )

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()  # 更新参数
        optimizer.zero_grad(set_to_none=True)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1:2d}，Loss={loss.item():.4f}")

weight_after = model.lm_head.weight.detach()

print(
    "\nLM Head 参数是否改变：",
    not torch.equal(weight_before, weight_after)
)