import torch
from model import GPT2
import tiktoken
import torch.nn
import torch.nn.functional as F
import os

tokenizer = tiktoken.get_encoding("gpt2")

my_model = GPT2(
    vocab_size=50257,
    n_blocks=12,
    n_heads = 12,
    n_embeddings = 768,
    max_seq_len = 1024,
)

if os.path.exists("model.pth"):
    print("Model found!")
    my_model.load_state_dict(torch.load("model.pth"))
else:
    print("No model found")

optimizer = torch.optim.AdamW(my_model.parameters(), lr=1e-4)


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
        chunk = self.toks[self.cur_pos:self.cur_pos+B*L+1]
        sample = chunk[:-1].view(B,L)
        truth = chunk[1:].view(B,L)
        self.cur_pos += B*L
        if self.cur_pos + B*L > len(self.toks):
            self.cur_pos = 0
        return sample,truth
B,L = 4,128
dl = dataLoader(B,L)


def train():
    for i in range(100):
        sample,truth = dl.get_batch()
        logits,loss = my_model.forward(sample,truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step: {i} Loss: {loss}")
        if i%10 == 0:
            torch.save(my_model.state_dict(),"model.pth")
            print("Model saved")