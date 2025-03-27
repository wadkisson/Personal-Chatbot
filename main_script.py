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
import time

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
max_steps = 1000
optimizer = model.set_optimimzers(lr=6e-4, weight_decay=0.1)

# Batch size and gradient accumulation setup
micro_batch_size = 16 # Smaller batch size that fits in memory
sequence_length = 1024
grad_accum_steps = 32  # Fixed number of gradient accumulation steps
effective_batch_size = micro_batch_size * grad_accum_steps
print(f"Effective batch size: {effective_batch_size}")

def load_tokens(file):
    tokens = np.load(file)
    tokens = torch.tensor(tokens,dtype = torch.long)
    return
class dataLoader:
    def __init__(self, batch_size, seq_len,split):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        root = "edu_fineweb10B"
        shard_list = os.listdir(root)
        shards = [shard for shard in shard_list if split in shard]
        shards = sorted(shards)
        shards = [os.path.join(root,shard) for shard in shards]
        self.shards = shards
        self.reset()
    def reset(self):
        self.current_shard = 0
        self.toks = load_tokens(self.shards[self.current_shard])
        self.cur_pos = self.batch_size * self.seq_len
    def get_batch(self):
        # Use instance variables instead of globals
        chunk = self.toks[self.cur_pos:self.cur_pos + self.batch_size*self.seq_len + 1]
        sample = chunk[:-1].view(self.batch_size, self.seq_len)
        truth = chunk[1:].view(self.batch_size, self.seq_len)
        self.cur_pos += self.batch_size * self.seq_len
        
        # Wrap around if we've reached the end
        if self.cur_pos + self.batch_size*self.seq_len + 1 > len(self.toks):
            self.current_shard = (self.current_shard + 1)%len(self.shards)
            self.toks = load_tokens(self.shards[self.current_shard])
            self.cur_pos = self.batch_size * self.seq_len
        return sample, truth

# Initialize data loader with micro batch size
dl_train = dataLoader(micro_batch_size, sequence_length,split = "train")
dl_val = dataLoader(micro_batch_size, sequence_length,split = "val")
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
    model.train()
    print("Starting training...")
    print(f"Micro batch size: {micro_batch_size}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    
    for i in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        total_loss = 0
            
        # Gradient accumulation loop
        for j in range(grad_accum_steps):
            sample, truth = dl_train.get_batch()
            sample, truth = sample.to(device), truth.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float16):
                logits, loss = model.forward(toks = sample, targets = truth)
            loss = loss / grad_accum_steps
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)*1000
        if i % 100 == 0:
            print(f"Step: {i} | Loss: {loss} | toks per second: {effective_batch_size/dt*1000}")
            model.eval()
            input = "Hello. My name is Jordan Belford. "
            tokens = tokenizer.encode(input)
            tokens = torch.tensor(tokens).unsqueeze(0).to(device)
            while tokens.size(1)<model.max_seq_len:
                with torch.no_grad():
                    logits, loss = model.forward(tokens)
                    logits = logits[:,-1,:]
                    probs = F.softmax(logits,-1)
                    probs,ind = torch.topk(probs,50,dim=-1)
                    i = torch.multinomial
                    col = torch.gather(ind,-1,i)
                    tokens = torch.cat((tokens,col),dim=1)
            tokens = tokens[0,:model.max_seq_len]
            text = tokenizer.decode(tokens)
            print(f"Here's a sample for ya: {text}")
            with torch.no_grad():
                dl_val.reset()
                val_loss = 0
                val_steps = 20
                for _ in range (val_steps):
                    sample, truth = dl_val.get_batch()
                    sample = sample.to(device)
                    truth = truth.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float16):
                        logits, loss = model.forward(toks = sample, targets = truth)
                    loss = loss/val_steps
                    val_loss += loss.detatch()
            print(f"VAL LOSS: {val_loss}")
            model.train()

train()