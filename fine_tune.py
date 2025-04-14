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
import pyautogui
import random

NUM_STEPS = 0
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
    model = torch.compile(model)

if os.path.exists("model_FINETUNE.pth"):
    print("Model found!")
    model.load_state_dict(torch.load("model_FINETUNE.pth"))
else:
    print("No model found")


# Training parameters
max_lr = 2e-5
min_lr = 2e-6
max_steps = 600
warmup = 60
optimizer = model.set_optimizers(lr=max_lr, weight_decay=0.1)
micro_batch_size = 16
sequence_length = 1024
grad_accum_steps = 32
effective_batch_size = micro_batch_size * grad_accum_steps
print(f"Effective batch size: {effective_batch_size}")

class TherapyDataLoader:
    def __init__(self, tokens_list, batch_size, seq_len):
        self.tokens_list = tokens_list
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.flat_tokens = torch.cat(tokens_list)
        self.usable_len = (len(self.flat_tokens) // (batch_size * seq_len)) * (batch_size * seq_len)
        self.flat_tokens = self.flat_tokens[:self.usable_len]
        self.flat_tokens = self.flat_tokens.view(batch_size, -1)
        self.reset()
    
    def reset(self):
        self.current_pos = random.randint(0, self.flat_tokens.size(1) - self.seq_len - 1)
    
    def get_batch(self):
        start = random.randint(0, self.flat_tokens.size(1) - self.seq_len - 1)
        chunk = self.flat_tokens[:, start:start + self.seq_len + 1]
        sample = chunk[:, :-1]
        truth = chunk[:, 1:]
        return sample, truth

def get_lr(i):
    if i + NUM_STEPS < warmup:
        return max_lr * (i+1+NUM_STEPS)/warmup
    if i + NUM_STEPS > max_steps:
        return min_lr
    decay = ((i + NUM_STEPS)-warmup)/(max_steps-warmup)
    assert decay >= 0 and decay <= 1
    coeff = 0.5 * (1+math.cos(math.pi*decay))
    return min_lr + coeff * (max_lr-min_lr)

# Load and prepare dataset
print("Loading dataset...")
dataset = load_dataset("jerryjalapeno/nart-100k-synthetic")["train"]

def format_conversation(example):
    convo = example["conversations"]
    formatted = "<|startoftext|>"
    for msg in convo:
        role = msg["from"].strip()
        text = msg["value"].strip()
        if role == "human":
            formatted += f" Patient: {text}"
        elif role == "gpt":
            formatted += f" Therapist: {text}"
    formatted += " <|endoftext|>"
    return formatted

print("Tokenizing conversations...")
formatted_conversations = [format_conversation(example) for example in dataset]
tokens_list = [torch.tensor(tokenizer.encode(full_convo, allowed_special={"<|endoftext|>", "<|startoftext|>"})) 
               for full_convo in formatted_conversations]

# Initialize data loader
dl_train = TherapyDataLoader(tokens_list, micro_batch_size, sequence_length)

def train():
    model.train()
    print("Starting training...")
    print(f"Micro batch size: {micro_batch_size}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    temperature = 1.0
    topk = 200
    
    for i in range(max_steps):
        pyautogui.click()
        t0 = time.time()
        optimizer.zero_grad()
        total_loss = 0.0
        
        # Gradient accumulation loop
        for _ in range(grad_accum_steps):
            sample, truth = dl_train.get_batch()
            sample, truth = sample.to(device), truth.to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float16):
                logits, loss = model.forward(toks=sample, targets=truth)
            loss = loss / grad_accum_steps
            total_loss += loss.detach()
            loss.backward()
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(i + NUM_STEPS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        print(f"Step: {i + NUM_STEPS} | Loss: {total_loss} | toks per second ={micro_batch_size*sequence_length/dt},lr={lr}")
        
        with open('finet_loss_history.txt', 'a') as f:
            f.write(f"{i + NUM_STEPS},{total_loss.item():.4f},{effective_batch_size/dt*1000:.2f},{lr:.6f}\n")
        
        if (i + NUM_STEPS) % 50 == 0:
            time.sleep(60)
            torch.save(model.state_dict(), "model_FINETUNE.pth")
            model.eval()
            
            input_text = "<|startoftext|>Patient: I'm going through a really tough time right now. Can I talk to you about my problems?Therapist:"
            tokens = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0).to(device)
            
            while tokens.size(1) < model.max_seq_len:
                with torch.no_grad():
                    logits, loss = model.forward(tokens)
                    logits = logits[:,-1,:]
                    logits = logits/temperature
                    probs = F.softmax(logits, dim=-1)
                    tk_probs, ind = torch.topk(probs, topk, dim=-1)
                    idx = torch.multinomial(tk_probs, 1)
                    col = torch.gather(ind, -1, idx)
                    tokens = torch.cat((tokens, col), dim=1)
                    
                    if col.item() == end_of_text_token:
                        break
            
            try:
                text = tokenizer.decode(tokens[0].tolist())
                print(f"Sample conversation:\n{text}")
            except Exception as e:
                print(f"Error in decoding: {e}")
            
            # Validation
            with torch.no_grad():
                val_loss_accum = 0.0
                val_steps = 20
                for _ in range(val_steps):
                    sample, truth = dl_train.get_batch()
                    sample, truth = sample.to(device), truth.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float16):
                        val_logits, val_loss = model.forward(toks=sample, targets=truth)
                    val_loss = val_loss/val_steps
                    val_loss_accum += val_loss.detach()
            
            print(f"VAL LOSS: {val_loss_accum}")
            with open("finet_val_samples.txt", "a") as f:
                f.write(f"\nStep {i + NUM_STEPS}:\n{text}\nVAL LOSS: {val_loss_accum:.4f}\n")
            
            model.train()

train()
