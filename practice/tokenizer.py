from pathlib import Path
from transformers import AutoTokenizer

# 本文件放在 minimind/practice/ 下。
# parent 是 practice，parent.parent 是 minimind 项目目录。
project_dir = Path(__file__).resolve().parent.parent

# 从项目自带的 model 目录加载分词规则和配置。
# local_files_only=True 表示只读取本地文件。
tokenizer = AutoTokenizer.from_pretrained(
    str(project_dir / "model"),
    local_files_only=True,
)

text = "PyTorch 找不到 GPU"

# 将文本编码成 ID 列表。
# 暂时关闭自动添加特殊 token，观察文本本身的分词结果。
ids = tokenizer.encode(text, add_special_tokens=False)

print("原始文本：", text)
print("token ID：", ids)
print("token 数量：", len(ids))
# ID 转成词表里的 token 表示。
print("词表 token：", tokenizer.convert_ids_to_tokens(ids))

# 将整个 ID 序列解码回文本。
print("解码文本：", tokenizer.decode(ids),"\n")

# 相同文本也可以直接编码为 PyTorch 张量。
encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
print(encoded,"\n")
print("input_ids：", encoded["input_ids"])
print("形状：", encoded["input_ids"].shape)
print("类型：", encoded["input_ids"].dtype)


# 在你装好依赖的环境里，从 minimind 目录执行。
# python practice/l05_tokenizer.py
# 预期结果是输出 ID 列表、分词数量、解码文本，以及形状为 (1, T) 的整数张量。T 是实际分词数量。

# 完成后，你需要能独立做到三件事。
# 1. 将输入改为 "learning" 或 "我喜欢学习"，解释字符数与 token 数为什么可能不同。
# 2. 说明 encode、`convert_ids_to_tokens`、`decode` 各返回什么。
# 3. 解释 return_tensors="pt" 为什么让输出多了一个 batch 维度。
# 以后按你指定的方式配合。需要你掌握书写的练习，我先说明目标，再在对话给带注释的代码；用于观察现象的辅助脚本，我直接写入文件并运行，再讲结果。 这次尚未创建文件。