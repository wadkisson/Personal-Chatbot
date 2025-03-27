#nan gpt repo i mention so much:
#https://github.com/karpathy/nanoGPT
import torch
import numpy as np
from model import GPT2
import tiktoken
import torch.nn
import torch.nn.functional as F
import os
import math
from datasets import load_dataset
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

torch.set_float32_matmul_precision('high')

model = GPT2(
    vocab_size = 50304,  # taken straight from nanoGPT, more efficient because its a power of 2
    n_blocks=12,
    n_heads = 12,
    n_embeddings = 768,
    max_seq_len = 1024,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("GPU found")
else:
    print("No GPU found")

model = model.to(device)
if device == 'cuda':
    model = torch.compile(model) #- use for gpu training

if os.path.exists("model.pth"):
    print("Model found!")
    model.load_state_dict(torch.load("model.pth"))
else:
    print("No model found")

# Training parameters
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup = 10
max_steps = 50
optimizer = model.set_optimimzers(lr=6e-4, weight_decay=0.1)

# Batch size and gradient accumulation setup
micro_batch_size = 16 # Smaller batch size that fits in memory
sequence_length = 1024
grad_accum_steps = 32  # Fixed number of gradient accumulation steps
effective_batch_size = micro_batch_size * grad_accum_steps
print(f"Effective batch size: {effective_batch_size}")


class dataLoader:
    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        with open('tinyS.txt','r') as f:
            text = f.read()
        self.text = tokenizer.encode(text)
        self.toks = torch.tensor(self.text)
        self.cur_pos = 0
        print(f"Dataset size: {len(self.toks)} tokens")
        
    def get_batch(self):
        # Use instance variables instead of globals
        chunk = self.toks[self.cur_pos:self.cur_pos + self.batch_size*self.seq_len + 1]
        if len(chunk) < self.batch_size*self.seq_len + 1:
            # If we don't have enough tokens, wrap around
            self.cur_pos = 0
            chunk = self.toks[self.cur_pos:self.cur_pos + self.batch_size*self.seq_len + 1]
            
        sample = chunk[:-1].view(self.batch_size, self.seq_len)
        truth = chunk[1:].view(self.batch_size, self.seq_len)
        self.cur_pos += self.batch_size * self.seq_len
        
        # Wrap around if we've reached the end
        if self.cur_pos + self.batch_size*self.seq_len > len(self.toks):
            self.cur_pos = 0
            
        return sample, truth

# Initialize data loader with micro batch size
dl = dataLoader(micro_batch_size, sequence_length)

def get_lr(i):
    if i < warmup:
        return max_lr * (i+1)/warmup
    if i > max_steps:
        return min_lr
    decay = (i-warmup)/(max_steps-warmup)
    assert decay >= 0 and decay <= 1
    coeff = 0.5 * (1+math.cos(math.pi*decay))
    return min_lr + coeff * (max_lr-min_lr)

def train():
    print("Starting training...")
    print(f"Micro batch size: {micro_batch_size}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    
    for i in range(max_steps):
        optimizer.zero_grad()
        total_loss = 0
            
        # Gradient accumulation loop
        for j in range(grad_accum_steps):
            sample, truth = dl.get_batch()
            print(f"Sample shape: {sample.shape}, Truth shape: {truth.shape}")
            sample, truth = sample.to(device), truth.to(device)
            print("moved to device")
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float16):
                print("autocast")
                logits, loss = model.forward(toks = sample, targets = truth)
                print("we got loss!")
            print("forward")
            loss = loss / grad_accum_steps
            print("loss")
            loss.backward()
            print("loss backward done!!!!!!!")
        print("backward")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print("clip grad norm")
        lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        print("step")
        if device == 'cuda':
            torch.cuda.synchronize()
        if i % 100 == 0:
            print(f"Step: {i} Loss: {loss:.4f}, LR: {lr:.2e}")
        if i % 1000 == 0:
            torch.save(model.state_dict(), "model.pth")
            print("Model saved")


train()