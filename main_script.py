#nan gpt repo i mention so much:
#https://github.com/karpathy/nanoGPT
import torch
from model import GPT2
import tiktoken
import torch.nn
import torch.nn.functional as F
import os
import math

tokenizer = tiktoken.get_encoding("gpt2")

torch.set_float32_matmul_precision('high')

my_model = GPT2(
    #vocab_size=50257,
    vocab_size = 50304,#taken straight from nanoGPT, more efficient because its a power of 2, 16, 32, etc.
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
my_model = my_model.to(device)
model = torch.compile(my_model)

if os.path.exists("model.pth"):
    print("Model found!")
    my_model.load_state_dict(torch.load("model.pth"))
else:
    print("No model found")

optimizer = torch.optim.AdamW(my_model.parameters(), lr=1e-4,betas=(0.9,0.95))


class dataLoader:
    def __init__(self,B,L):
        self.B = B
        self.L = L

        with open('tinyS.txt','r') as f:
            text = f.read()
        self.text = tokenizer.encode(text)
        self.toks = torch.tensor(self.text)
        self.cur_pos = 0
    def get_batch(self):
        B, L = self.B,self.L
        #B,L = B.to(device),L.to(device)
        chunk = self.toks[self.cur_pos:self.cur_pos+B*L+1]
        sample = chunk[:-1].view(B,L)
        truth = chunk[1:].view(B,L)
        self.cur_pos += B*L
        if self.cur_pos + B*L > len(self.toks):
            self.cur_pos = 0
        return sample,truth
B,L = 4,128
dl = dataLoader(B,L)


#this concept of this lr scheduler is taken from the gpt2 paper
#but, credit where credit is due, I got the implementation from the nanoGPT repo
warmup_steps = 10
max_steps = 50
max_lr = 1e-4
min_lr = 1e-6

def update_lr(iteration):
    if iteration < warmup_steps:
        return max_lr * (iteration+1) / warmup_steps
    if iteration > max_steps:
        return min_lr
    ratio = (iteration - warmup_steps) / (max_steps - warmup_steps)
    coefficient = 0.5 * (1 + math.cos(math.pi * ratio))
    return min_lr + coefficient * (max_lr - min_lr)



def train():
    for i in range(max_steps):
        sample,truth = dl.get_batch()
        sample,truth = sample.to(device),truth.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type = device,dtype = torch.bfloat16):
            logits,loss = my_model.forward(sample,truth)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), 1.0)#gradient clipping (prevents exploding grads)
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        print(f"Step: {i} Loss: {loss}")
        if i%10 == 0:
            torch.save(my_model.state_dict(),"model.pth")
            print("Model saved")
train()