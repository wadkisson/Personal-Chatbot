import torch
import torch.nn as nn
import math
from torch.nn import functional as F
class Feed_forward_block(nn.Module):
    def __init__(self,n_embeddings):
        super().__init__()
        self.FF = nn.Linear(n_embeddings,4 * n_embeddings)
        self.proj = nn.Linear(4 * n_embeddings,n_embeddings)
        self.GELU = nn.GELU()
    def forward(self,x):
        x = self.FF(x)
        x = self.GELU(x)
        x = self.proj(x)
        return x
class Attention(nn.Module):
    def __init__(self, n_embeddings,n_heads):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.attn_linear = nn.Linear(self.n_embeddings, 3 * self.n_embeddings,)
        self.attn_proj = nn.Linear(self.n_embeddings, self.n_embeddings)
        self.n_heads = n_heads
    def forward(self, x):
        batches, seq_len, _ = x.size()
        Q,K,V = self.attn_linear(x).split(self.n_embeddings,dim=2)
        Q = Q.view(batches, seq_len, self.n_heads, self.n_embeddings // self.n_heads).transpose(1,2)
        K = K.view(batches, seq_len, self.n_heads, self.n_embeddings // self.n_heads).transpose(1,2)
        V = V.view(batches, seq_len, self.n_heads, self.n_embeddings // self.n_heads).transpose(1,2)
        x = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))
        #x = x.masked_fill(self.bias[:,:,:seq_len,:seq_len] == 0, float('-inf'))
        x = F.softmax(x, dim=-1)
        x = x @ V 
        x = x.transpose(1, 2).contiguous().view(batches, seq_len, self.n_embeddings)
        return x
class Transformer_Block(nn.Module):
    def __init__(self,n_embeddings,n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)
        self.attention = Attention(n_embeddings,n_heads)
        self.Feed = Feed_forward_block(n_embeddings)
    def forward(self,x):
        x = x + self.attention(self.ln1(x))
        x = x + self.Feed(self.ln2(x))
        return x
class GPT2(nn.Module):
    def __init__(self,n_embeddings,vocab_size,max_seq_len,n_blocks,n_heads):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.max_seq_len = max_seq_len
        self.wte = nn.Embedding(vocab_size, n_embeddings)
        self.wpe = nn.Embedding(max_seq_len, n_embeddings)
        self.final_norm = nn.LayerNorm(n_embeddings)
        self.blocks = nn.ModuleList([Transformer_Block(n_embeddings,n_heads) for _ in range(n_blocks)])
        self.final_projection = nn.Linear(n_embeddings,vocab_size)
    def forward(self,toks,targets=None):
        B, seq_len = toks.size()
        token_positions = torch.arange(0,seq_len)
        token_embedding = self.wte(toks)
        position_embedding = self.wpe(token_positions)
        x = token_embedding + position_embedding
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.final_projection(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits, loss