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
import tiktoken
import time
import pyautogui
import random

NUM_STEPS = 19050
tokenizer = tiktoken.get_encoding("gpt2")
end_of_text_token = tokenizer._special_tokens['<|endoftext|>']

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

if os.path.exists("model2.pth"):
    print("Model found!")
    model.load_state_dict(torch.load("model2.pth"))
else:
    print("No model found")

# Training parameters
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup = 715
max_steps = 19050
optimizer = model.set_optimimzers(lr=max_lr,weight_decay=0.1)
# Batch size and gradient accumulation setup
micro_batch_size = 16 # Smaller batch size that fits in memory
sequence_length = 1024
grad_accum_steps = 32  # Fixed number of gradient accumulation steps
effective_batch_size = micro_batch_size * grad_accum_steps
print(f"Effective batch size: {effective_batch_size}")

def load_tokens(file):
    tokens = np.fromfile(file,dtype= np.uint16)
    tokens = torch.tensor(tokens,dtype = torch.long)
    return tokens

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
        assert len(shards) > 0, f"No shards found in split {split}"
        self.reset()
        
    def reset(self):
        self.current_shard = 0
        random.shuffle(self.shards)
        self.toks = load_tokens(self.shards[self.current_shard])
        offset = len(self.toks)-self.batch_size*self.seq_len-1
        self.cur_pos = random.randint(0,max(1,offset))
    def get_batch(self):
        B,T = self.batch_size, self.seq_len
        chunk = self.toks[self.cur_pos:self.cur_pos + B*T + 1]
        sample = (chunk[:-1]).view(B, T)
        truth = (chunk[1:]).view(B, T)
        self.cur_pos += B*T
        
        # Wrap around if we've reached the end
        if self.cur_pos + B*T + 1 > len(self.toks):
            self.current_shard = (self.current_shard + 1)%len(self.shards)
            self.toks = load_tokens(self.shards[self.current_shard])
            self.cur_pos = B*T
        return sample, truth

# Initialize data loader with micro batch size
dl_train = dataLoader(micro_batch_size, sequence_length,split = "train")
dl_val = dataLoader(micro_batch_size, sequence_length,split = "val")

def get_lr(i):
    if i + NUM_STEPS< warmup:
        return max_lr * (i+1+NUM_STEPS)/warmup
    if i + NUM_STEPS> max_steps:
        return min_lr
    decay = ((i + NUM_STEPS)-warmup)/(max_steps-warmup)
    assert decay >= 0 and decay <= 1
    coeff = 0.5 * (1+math.cos(math.pi*decay))
    return min_lr + coeff * (max_lr-min_lr)

def train():
    model.train()
    print("Starting training...")
    print(f"Micro batch size: {micro_batch_size}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    temperature = 1.0 # greater temp = more randomness in generation.
    topk = 200 # oops, meant to be 200.
    
    for (i) in range(max_steps):
        pyautogui.click()
        t0 = time.time()
        optimizer.zero_grad()
        total_loss = 0.0
        rep_penalty = 1.2 #add a repitition penalty for the model
        # Gradient accumulation loop
        for _ in range(grad_accum_steps):
            sample, truth = dl_train.get_batch()
            sample, truth = sample.to(device), truth.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float16):
                logits, loss = model.forward(toks = sample, targets = truth)
            loss = loss / grad_accum_steps
            total_loss+=loss.detach()
            loss.backward()
     
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(i + NUM_STEPS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0#total time in seconds
        print(f"Step: {i + NUM_STEPS} | Loss: {total_loss} | toks per second ={micro_batch_size*sequence_length/dt},lr={lr}")
        with open ('loss_history.txt','a') as f:
                f.write(f"{i + NUM_STEPS},{total_loss.item():.4f},{effective_batch_size/dt*1000:.2f},{lr:.6f}\n")

        if (i + NUM_STEPS)% 50 == 0:
            time.sleep(60)
            torch.save(model.state_dict(),"model2.pth")
            model.eval()
            with torch.no_grad():
                dl_val.reset()
                val_loss_accum = 0.0
                val_steps = 20
                for _ in range (val_steps):
                    sample, truth = dl_val.get_batch()
                    sample = sample.to(device)
                    truth = truth.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float16):
                        val_logits, val_loss = model.forward(toks = sample, targets = truth)
                    val_loss = val_loss/val_steps
                    val_loss_accum += val_loss.detach()
            print(f"VAL LOSS: {val_loss_accum}")
            with open("val_loss_history.txt","a") as f:
                f.write(f"{i + NUM_STEPS},{val_loss_accum:.4f}\n")

            input = "The causes of World War I were complex, involving alliances,"
            tokens = tokenizer.encode(input)
            tokens = torch.tensor(tokens).unsqueeze(0).to(device)
            while tokens.size(1)<model.max_seq_len:
                with torch.no_grad():
                    logits, loss = model.forward(tokens)
                    logits = logits[:,-1,:]
                    for t in set(tokens[0].tolist()):
                        logits[0,t]/=rep_penalty
                    logits = logits/temperature
                    probs = F.softmax(logits,dim=-1)
                    tk_probs,ind = torch.topk(probs,topk,dim=-1)
                    idx = torch.multinomial(tk_probs,1)
                    col = torch.gather(ind,-1,idx)
                    tokens = torch.cat((tokens,col),dim=1)
            tokens = tokens[0,:model.max_seq_len].tolist()
            
            try:
                text = tokenizer.decode(tokens)
                print(f"Here's a sample for ya: {text}")
                with open("samples.txt","a") as f:
                    f.write(f"STEP: {i+NUM_STEPS}\n{text}\n")
            except Exception as e:
                print(f"Error in decoding: {e}")
                with open("samples.txt","a") as f:
                    f.write(f"Error decoding these the tokens on step {i + NUM_STEPS}\n")
           
            model.train()

train()