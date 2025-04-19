"""
Simple launch using:
    python optimised-gpt2-train.py
DPP Launch using:
    torchrun --standalone --nproc_per_node=8 optimised-gpt2-train.py
"""

from dataclasses import dataclass
import inspect
import numpy as np
import math
import os
import tiktoken
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.nn import functional as F

@dataclass # adds all the dunder stuff to the class
class GPTConfig:
    block_size: int = 1024 # Context length
    vocab_size: int = 50257 # Word dictionary size
    n_layer: int = 12 # Number of layers (blocks)
    n_head: int = 12 # Number of heads
    n_embd: int = 768 # Embedding dimension

ddp = int(os.environ.get('RANK', -1)) != -1 # Check if multiple GPUs are available
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    if master_process:
        print(f"Running with DDP on {ddp_world_size} GPUs")
else:
    ddp_rank = 0
    ddp_world_size = 1
    ddp_local_rank = 0
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Running on device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

if master_process:
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    file_count = 0
    while os.path.exists(os.path.join(log_dir, f"log_{file_count}.txt")):
        file_count += 1
    log_file = os.path.join(log_dir, f"log_{file_count}.txt")
    # Can be opened in w to clear file first
    with open(log_file, "a") as f:
        pass

    with open(log_file, "a") as f:
        f.write(f"Running on device: {device}\n")
    with open(log_file, "a") as f:
        f.write(f"Running with DDP on {ddp_world_size} GPUs\n")

def load_tokens(filename):
    np_tensor = np.load(filename)
    np_tensor = np_tensor.astype(np.int32)
    pt_tensor = torch.tensor(np_tensor, dtype=torch.long)
    return pt_tensor

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ["train", "val"]
        
        # Get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [shard for shard in shards if split in shard]
        shards = sorted(shards)
        shards = [os.path.join(data_root, shard) for shard in shards]
        self.shards = shards
        
        assert len(self.shards) > 0, "No shards found"
        if master_process:
            print(f"Found {len(self.shards)} shards for {split} split")
        self.reset()


    def get_next_batch(self):
        buf = self.tokens[self.current_index : self.current_index + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        
        # Update the current index for the next batch based on process
        self.current_index += self.B * self.T * self.num_processes

        # Reset when out of bounds
        if self.current_index + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_index = self.B * self.T * self.process_rank
        return x, y

    def reset(self):
        # State init at shard 0
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # Calculate the starting index based on offset using process rank
        self.current_index = self.B * self.T * self.process_rank

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
        self.c_proj.CUSTOM_SCALING_INIT = 1
        
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

        # Efficient attention calculation using Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        self.mlp = MLP(config)
        
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
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'CUSTOM_SCALING_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def set_optimiser(self, weight_decay, learning_rate, device_type):
        # Fetch all parameters that require grad
        params = {pn: p for pn, p in self.named_parameters()}
        params = {pn: p for pn, p in params.items() if p.requires_grad}
        
        # Create groups for parameters that require weight decay
        decay_params = [p for n, p in params.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in params.items() if p.dim() < 2]
        grouped_params = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        if master_process:
            print(f"Number of decay params: {num_decay_params}")
            print(f"Number of no decay params: {num_no_decay_params}")
        
        # Create AdamW optimiser with fused AdamW if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        if master_process:
            print(f"Using fused AdamW: {use_fused}")
        optimiser = torch.optim.AdamW(grouped_params,
                                        lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimiser
    
    def forward(self, text, targets=None):
        # Raw data input of b context length sized sequences of raw text
        B, T = text.size()
        assert T <= self.config.block_size, "Sequence length too high"
        # Create numbers from 0 to T and move to the correct device
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

enc = tiktoken.get_encoding("gpt2")

# Enable humongous batch size on GPU !!!
batch_size, context_length = 64, 1024

# Enable small batch size on CPU
training_loader = DataLoader(batch_size, context_length, ddp_rank, ddp_world_size, split="train")
validation_loader = DataLoader(batch_size, context_length, ddp_rank, ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device);
# Compile the model (on GPU) to remove Python overhead !!!
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# Set up learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # Same ratio of warmup as GPT3
max_steps = 19073 # Exactly 1 epoch over the 10B token dataset
def get_lr(step):
    # Linear warmup to max_lr for warmup steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # Min learning rate after max steps
    if step > max_steps:
        return min_lr

    # Cosine decay for the rest of the steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    # Coeff starts at 1 and goes to 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   
    return min_lr + (max_lr - min_lr) * coeff

# Set up gradient accumulation
total_batch_tokens = 524288

assert total_batch_tokens % (batch_size * context_length * ddp_world_size) == 0, \
    "Total batch size must be divisible by mini batch size"

grad_accum_steps = total_batch_tokens // (batch_size * context_length * ddp_world_size)

# Set up optimiser
optimiser = raw_model.set_optimiser(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

for step in range(max_steps):
    t0 = time.time()
    
    # Validate every 100 steps
    if step % 100 == 0:
        model.eval()
        validation_loader.reset()
        with torch.no_grad():
            accumulated_val_loss = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = validation_loader.get_next_batch()
                # Move tensors to the correct device
                x = x.to(device)
                y = y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16): 
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                accumulated_val_loss += loss.detach()
        if ddp:
            torch.distributed.all_reduce(accumulated_val_loss, op=torch.distributed.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {accumulated_val_loss.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"Step {step}: val_loss={accumulated_val_loss.item():.4f}\n")
    
    # Save every 5000 steps or at the end of the training
    if master_process and step > 0 and (step % 5000 == 0 or (step == max_steps - 1)):
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            'model': raw_model.state_dict(),
            'config': raw_model.config,
            'step': step,
            'optimiser': optimiser.state_dict(),
            'rng_seed': 42
        }
        if master_process:
            torch.save(checkpoint, checkpoint_path)
            
    # Training loop
    model.train()
    optimiser.zero_grad()
    accumulated_loss = 0.0
    # Accumulate gradient prior to optimiser step
    for mini_step in range(grad_accum_steps):
        x, y = training_loader.get_next_batch()
        # Move tensors to the correct device
        x = x.to(device)
        y = y.to(device)
        # Autocast forward pass parameters to BF16 (Enable when CUDA) !!!
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16): 
        # MPS only supports FP16 not BF16 :(
        # with torch.autocast(device_type=device, dtype=torch.float16): 
            logits, loss = model(x, y)

        # Ensure that loss normalisation is still present after accumulation
        loss = loss / grad_accum_steps
        accumulated_loss += loss.detach()
        # Only sync gradients on the last mini batch
        if ddp:
            model.require_backward_grad_sync = (mini_step == grad_accum_steps - 1)
        loss.backward()
    # Ensure that the displayed loss is averaged across all the GPUs
    if ddp:
        torch.distributed.all_reduce(accumulated_loss, op=torch.distributed.ReduceOp.AVG) 

    # Clip the gradient norms to a maximum of 1
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Determine and set the learning rate for the current step
    lr = get_lr(step)
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr
    optimiser.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()

    # measure time taken for each step
    dt = (t1 - t0)
    # Calculate tokens processed across all GPUs
    tokens_processed = training_loader.B * training_loader.T * grad_accum_steps * ddp_world_size
    # measure throughput of tokens/sec at each step
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"Step {step}: train_loss={accumulated_loss.item():.6f} lr={lr:.6f} norm={norm:.4f} t={dt*1000:.2f}ms tok/sec={tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"Step {step}: train_loss={accumulated_loss.item():.6f} lr={lr:.6f} norm={norm:.4f} t={dt*1000:.2f}ms tok/sec={tokens_per_sec:.2f}\n")

# Save final model
if master_process:
    final_checkpoint_path = os.path.join(log_dir, f"model_10B_toks_final.pt")
    final_checkpoint = {
        'model': raw_model.state_dict(),
        'config': raw_model.config,
        'step': step,
        'optimiser': optimiser.state_dict(),
        'rng_seed': 42
    }
    torch.save(final_checkpoint, final_checkpoint_path)

if ddp:
    destroy_process_group()
