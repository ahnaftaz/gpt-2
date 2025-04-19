from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass # adds all the dunder stuff to the class
class GPTConfig:
    block_size: int = 1024 # Context length
    vocab_size: int = 50257 # Word dictionary size
    n_layer: int = 12 # Number of layers (blocks)
    n_head: int = 12 # Number of heads
    n_embd: int = 768 # Embedding dimension

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"Detected device: {device}")

import tiktoken

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding('gpt2')
        with open('data/sample_input.txt', 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.tokens = self.tokens.to(device)
        print("num tokens in dataset:", len(self.tokens))
        print("batches to 1 epoch:", len(self.tokens) // (self.B * self.T))

        self.current_index = 0

    def get_next_batch(self):
        buf = self.tokens[self.current_index : self.current_index + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_index += self.B * self.T

        # Reset when out of bounds
        if self.current_index + (self.B * self.T + 1) > len(self.tokens):
            self.current_index = 0
        return x, y

    def reset(self):
        self.current_index = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Confirm dimensions will be evenly distributed among all the heads
        assert config.n_embd % config.n_head == 0
        # Create the key, query and value projections in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Following info will need to be stored since forward pass needs to 
        # separate the abomination above
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        # The masked filter but idk why OpenAI called it bias.
        # Resised to fit the above abomination
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1, config.block_size, config.block_size))
        # Linear projection out of the Attention block
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
    def forward(self, x):
        # Batch, time and channel of data (batch size, sequence length & emb dim)
        B, T, C = x.size()
        # Calculate the qkv value combined in the shape (B,T,3 * C)
        qkv = self.c_attn(x)
        # Split n_embd size bits out for k, q, v along the channel dimension
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Reshape each tensor to have the heads in dim=2 and then steal the weights for
        # those heads by taking the values from the embedding dimension.
        # Transpose the sequence length and head size so that the affinity calculation
        # is completed on the sequence length * head size face of the matrices
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1) # (B, n_heads, T, T)
        y = att @ v # (B, n_heads, T, T) @ (B, n_heads, T, head_size) = (B, n_heads, T, head_sze)
        # Re-orient tensor to shape (B, T, n_heads, head_size), followed by
        # Concatenation via the view method
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Contiguous method ensures that the entire data is stored in a nice way
        # such that no memory errors can occur when we do the concatenating.
        # This is more important in our gpt-2 size model because our memory usage
        # is high enough that our OS may split the memory to different places
        
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Expanding the dimensions of the data to let some 
        # computation occur
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # gelu activation uses a tanh to approximate the real function
        # in the GPT2 due to an error in PyTorch but we need the exact
        # same stuff to import properly so we gotta suffer same way
        self.gelu = nn.GELU(approximate="tanh")
        # Projecting the data back into normal dimensions
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        
        # Adding a custom flag to the model to flag our scaling
        self.c_proj.CUSTOM_SCALING_INIT = 1
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = (config)
        
    def forward(self, x):
        # Residual skip connection for the attention block with pre-norm
        x = x + self.attn(self.ln_1(x))
        # Residual skip connection for the MLP block with pre-norm
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            # dictionary of token embedding weights (weight of token embeddings)
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # dictionary of positional embedding weights (weight of positional embeddings)
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Attention blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final Layer norm (since pre-norm doesn't touch final block output)
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        # Linear projection out of the Attention block
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight sharing between wte and lm_head 
        self.transformer.wte.weight = self.lm_head.weight
        
        # Apply weight initialisation
        self.apply(self._init_weights) 
    
    def _init_weights(self, module):
        # Initialise the weights of the model using the same method as in GPT2
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'CUSTOM_SCALING_INIT'):
                std = (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def forward(self, text, targets=None):
        # Raw data input of b context length sized sequences of raw text
        B, T = text.size()
        assert T <= self.config.block_size, "Sequence length too high"
        # Create numbers from 0 to T
        pos = torch.arange(0, T, dtype=torch.long, device=text.device)
        # Fetch positional and token embeddings for all the values in text
        pos_emb = self.transformer.wpe(pos) # of size (T, n_embd)
        tok_emb = self.transformer.wte(text) # of size (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # of size (B, T, vocab_size)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None
        return logits, loss
    
import time

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.manual_seed(42)

# Enable humongous batch size on GPU
# training_loader = DataLoader(B=16, T=1024)

# Enable small batch size on CPU
training_loader = DataLoader(B=8, T=1024)

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.to(device);

optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=25, gamma=0.1)
steps = 50
for i in range(steps):
    t0 = time.time()
    x, y = training_loader.get_next_batch()
    optimiser.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimiser.step()
    t1 = time.time()
    # measure time taken for each step
    dt = (t1 - t0) * 1000
    # measure throughput of tokens/sec at each step
    tokens_per_sec = (training_loader.B * training_loader.T) / (t1 - t0)
    # scheduler.step()
    print(f"Step {i+1}: Loss={loss.item():.4f} Time {dt:.2f}ms Tokens/sec={tokens_per_sec:.2f}")