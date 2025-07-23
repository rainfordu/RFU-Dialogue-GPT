# training configuration for a small-scale prompt-based GPT model

out_dir = 'out-wiki-prompt'
eval_interval = 250
eval_iters = 200
log_interval = 10

# save checkpoint only when validation improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'wiki-prompt'
wandb_run_name = 'gpt-mini-prompt'

# point to your dataset (you must've prepared it with prepare.py or manually)
dataset = 'wiki'  # 你需要把 input.txt 放在 data/wiki_prompt/input.txt 下
gradient_accumulation_steps = 2
batch_size = 32
block_size = 512  # 提升 context window 支持更长问答对

# model parameters (small size, good for finetune)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# optimizer settings
learning_rate = 1e-3
max_iters = 6000
lr_decay_iters = 6000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# system
device = 'cuda'  # 如果你想强制 CPU，可改为 'cpu'
compile = False  # Torch 2.0 compile 支持（小模型禁用节省时间）

# tokenizer
tokenizer = 'bpe'  # 你需要配合 tokenizer 来设置 prepare_bpe.py 版本
vocab_size = None  # 会自动从 tokenizer 生成的 meta.pkl 获取
