import torch
from model import GPT, GPTConfig
import tiktoken
import os

# --------------------------- 配置 ---------------------------
checkpoint_path = 'out/ckpt.pt'  # 你的模型权重路径
meta_path = 'data/wiki/meta.pkl'  # 你的meta.pkl路径

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------------------- 加载 meta.pkl ---------------------------
import pickle
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
tokenizer_name = meta.get('tokenizer', 'gpt2_bpe')

# --------------------------- 初始化 tokenizer ---------------------------
enc = tiktoken.get_encoding("gpt2")  # 这里默认gpt2分词，你可以改成 tokenizer_name

# --------------------------- 加载模型 ---------------------------
model_args = dict(
    n_layer=12,   # 需要和你训练时保持一致
    n_head=12,
    n_embd=768,
    vocab_size=vocab_size,
    block_size=1024,
    dropout=0.0,
    bias=False,
)

model = GPT(GPTConfig(**model_args))
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# --------------------------- 生成函数 ---------------------------
@torch.no_grad()
def generate_text(prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    model.eval()
    context_tokens = enc.encode(prompt)
    if len(context_tokens) > model.config.block_size:
        context_tokens = context_tokens[-model.config.block_size:]  # 截断
    input_ids = torch.tensor([context_tokens], dtype=torch.long, device=device)

    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
       # eos_token_id=None,
       # do_sample=True,
    )
    generated_tokens = generated[0].tolist()
    # 解码生成的所有token（包含prompt），这里去掉prompt只返回新生成部分
    output_text = enc.decode(generated_tokens[len(context_tokens):])
    return output_text

# --------------------------- 交互循环 ---------------------------
print("欢迎使用 RFU-Dialogue-GPT 对话。输入 'exit' 退出。")
while True:
    prompt = input("你说: ")
    if prompt.strip().lower() == 'exit':
        break
    response = generate_text(prompt)
    print("机器人: " + response)
