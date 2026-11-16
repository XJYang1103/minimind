import os
import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

"""
prompt        问题字符串
    ↓ 加入 conversation
conversation  消息列表
    ↓ apply_chat_template
inputs        模板字符串
    ↓ tokenizer(...)
inputs        包含 ID、mask 张量的对象
    ↓ model.generate(...)
generated_ids 输入＋回答的 ID 张量
    ↓ 切掉输入部分，再 decode
response      回答字符串

"""

# 加载 Tokenizer，model
def init_model(args):
    # 用model\tokenizer.json来创建tokenizer，args.load_from默认为model\
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    #创建model
    #1.命令行手动提供 结构配置，权重路径
    #是检查路径最后一级名称是否恰好等于 "model"
    if os.path.basename(os.path.normpath(args.load_from)) == 'model':
        #MiniMindConfig定义配置，MiniMindForCausalLM创建Embedding、Attention、FFN 等模块及其参数
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''

        # 加载权重
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'       
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

        if args.lora_weight != 'None':
            apply_lora(model, rank=args.lora_rank)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}{moe_suffix}.pth')
    #2.库从文件读取配置并组织加载，在目录里指定好配置和权重
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--lora_rank', default=16, type=int, help="LoRA rank，需与训练时一致（默认16）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)

    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
#     streamer 用于模型生成过程中逐步显示文本，不用等整段回答生成完才打印。
#   - tokenizer 负责把生成的 ID 解码成文字。
#   - skip_prompt=True 跳过输入问题部分。
#   - skip_special_tokens=True 隐藏角色边界等特殊 token。
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 创建迭代器，每次for循环就向后迭代一次

    # 自动模式直接遍历 prompts。
    # 手动模式，iter迭代的执行input，每次输入作为下一个问题，直到输入''
    # iter(可调用对象, 停止值)  一旦iter检测到lambda: input('💬: ')返回的是''时，就停止迭代
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '') 
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        # historys = 2 → 保留最近 2 条消息，通常是一问一答
        conversation = conversation[-args.historys:] if args.historys else []  # 
        conversation.append({"role": "user", "content": prompt})
        # 预训练通常让模型学习“根据前文预测下一个 token”。因此，这里采用文本续写形式
        # 这是加载“预训练阶段得到的权重”进行推理。inputs就是让模型从当前prompt继续预测下一个token
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        #对话分支 ：历史和当前消息 + 角色边界 → 让模型接着以助手身份回答
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        """
        假设前面经过apply_chat_template，得到：
        inputs = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"  字符串

        - tokenizer会把字符串分词，变成序号。
        - return_tensors="pt"：把序号序列变成tensor形式
        - truncation=True：是否截断

        然后返回inputs=
        {
            "input_ids": tensor([
                [1, 25, 10, 301, 2, 10, 1, 26, 10]  #模型接收的 token 编号
            ]),
            "attention_mask": tensor([
                [1, 1, 1, 1, 1, 1, 1, 1, 1]   #	标识有效位置
            ])
        }

        - .to(args.device) 就是把inputs里面的内容放到device里，比如说gpu，但是这个inputs对象还是在cpu

        """
        # 填充好的模版字符串
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        # 生成回答
        print('🧠: ', end='')
        st = time.time()
        # 假设输入的问题是 20 个token，实际新增 8 个回答token。
        # 返回 generated_ids.shape = (1, 28)   [原输入的 20 个 ID | 新增回答的 8 个 ID]
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        # generated_ids[0]是取generated_ids(1, 28)的第一行数据
        # [len(inputs["input_ids"][0]):]从20开始取，只取那8条回答
        # 用decode把序号解码成文字token
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print()