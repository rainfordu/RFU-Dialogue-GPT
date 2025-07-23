RFU-Dialogue-GPT is based on nanoGPT（https://github.com/karpathy/nanoGPT）
Install:
    pip install torch numpy transformers datasets tiktoken wandb tqdm
Prepare:
    Add input.txt into data/wiki
    python data/wiki/prepare.py
Train:
    python train.py config/trainwiki.py
Sample:
    python sample.py
