import os
import pickle
import tiktoken
import numpy as np

# 路径设置
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
train_bin_path = os.path.join(os.path.dirname(__file__), 'train.bin')
val_bin_path = os.path.join(os.path.dirname(__file__), 'val.bin')
meta_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')

# 读取原始文本数据
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"Length of dataset in characters: {len(data):,}")

# 拆分训练集与验证集（90% / 10%）
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# 使用 GPT-2 的 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")  # 可替换为其他编码器
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
vocab_size = enc.n_vocab

print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")
print(f"Vocab size: {vocab_size}")

# 保存为 .bin 文件
np.array(train_ids, dtype=np.uint16).tofile(train_bin_path)
np.array(val_ids, dtype=np.uint16).tofile(val_bin_path)

# 保存元信息，用于模型加载 tokenizer
meta = {
    'vocab_size': vocab_size,
    'tokenizer': 'gpt2_bpe',  # 自定义名称
}
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

print("✅ BPE 数据准备完毕：train.bin, val.bin 和 meta.pkl")
