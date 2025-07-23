import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64" #减少内存碎片

import time
import math
import pickle
from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler   #Enable AMP for ACCELERATION
scaler = GradScaler()                             #Enable AMP for ACCELERATION

import matplotlib.pyplot as plt
train_losses = []
val_losses = []                                   #Save Loss Curve


import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
import tiktoken

# --------------------------- Configuration ---------------------------
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'


wandb_log = False
wandb_project = 'bpe-gpt'
wandb_run_name = 'bpe_run'

# Data config
dataset = 'wiki'
gradient_accumulation_steps = 40
batch_size = 32
block_size = 1024

# Model config
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# Optimizer config
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# DDP/system
backend = 'nccl'
device = 'cuda'
##dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16

compile = True

# --------------------------- DDP Init ---------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# --------------------------- Setup ---------------------------
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --------------------------- Tokenizer ---------------------------
enc = tiktoken.get_encoding("gpt2")
def encode(text): return enc.encode_ordinary(text)
def decode(tokens): return enc.decode(tokens)
vocab_size = enc.n_vocab

# --------------------------- Dataset ---------------------------
def get_batch(split):
    data = np.memmap(os.path.join('data', dataset, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# --------------------------- Model Init ---------------------------
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
model = GPT(GPTConfig(**model_args)).to(device)

#import torch                                                    #ACCELERATION
#model = torch.compile(model)  # Accelerated graph compilation   #ACCELERATION #MemoryInsufficient


iter_num = 0
best_val_loss = 1e9
if init_from == 'resume':
    checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    model.load_state_dict(checkpoint['model'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# --------------------------- Helpers ---------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx: 
                    _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name)

# --------------------------- Training Loop ---------------------------
X, Y = get_batch('train')
t0 = time.time()
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
       #losses = estimate_loss()
       #print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
       losses = estimate_loss()
       train_loss = losses['train']
       val_loss = losses['val']
       train_losses.append(train_loss)
       val_losses.append(val_loss)
       print(f"iter {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")  #Save Loss Curve
 
       if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

######for micro_step in range(gradient_accumulation_steps):
######       if ddp:
######            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
  ######      with ctx:
  ######          logits, loss = model(X, Y)
  ######          loss = loss / gradient_accumulation_steps
  ######      X, Y = get_batch('train')
   ######     scaler.scale(loss).backward()
        
    for micro_step in range(gradient_accumulation_steps):
       X, Y = get_batch('train')  # 放循环开头，每步都拿新数据

       if ddp:
        model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

       with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()      #Enable AMP for ACCELERATION


    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    dt = time.time() - t0
    t0 = time.time()
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

 # 新增：每50步强制保存一次checkpoint
    if iter_num > 0 and iter_num % 50 == 0 and master_process:
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, os.path.join(out_dir, f'ckpt.pt'))
        print(f"Forced checkpoint saved at iteration {iter_num}")
    
    iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

# 保存 loss 曲线图
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Evaluation Step")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid()
plt.savefig("loss_curve.png")
print("✅ Loss curve saved to loss_curve.png")            #Save Loss Curve

