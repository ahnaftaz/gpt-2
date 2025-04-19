from dataclasses import dataclass
import inspect
import numpy as np
import os
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

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

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.CUSTOM_SCALING_INIT = 1
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
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
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights) 
    
    def _init_weights(self, module):
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
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
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
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        if master_process:
            print(f"Using fused AdamW: {use_fused}")
        optimiser = torch.optim.AdamW(
            grouped_params,
            lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimiser
    
    def forward(self, text, targets=None):
        B, T = text.size()
        assert T <= self.config.block_size, "Sequence length too high"
        pos = torch.arange(0, T, dtype=torch.long, device=text.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(text)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint['config']
    model = GPT(model_config)
    
    state_dict = checkpoint['model']
    new_state_dict = {}
    for key in state_dict:
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    model.load_state_dict(new_state_dict)
    return model

def generate(
    model,
    enc,
    prompt="",
    temperature=0.8,
    top_k=200,
    max_context_length=1000,
    max_new_tokens=500  # <--- NEW: limit total tokens to generate
):
    """
    Token-by-token generator. Yields strings (tokens) one at a time.
    
    :param model: The language model
    :param enc: The tokenizer
    :param prompt: Initial text prompt
    :param temperature: Sampling temperature
    :param top_k: Top-k sampling
    :param max_context_length: The maximum number of tokens we keep in context
    :param max_new_tokens: The max number of tokens to generate (avoid indefinite loop)
    """
    model.eval()
    if prompt:
        context = torch.tensor(
            enc.encode(prompt, allowed_special={"<|endoftext|>"}),
            dtype=torch.long
        ).unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long)

    context = context.to(next(model.parameters()).device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if context.size(1) > max_context_length:
                # Keep only the last max_context_length tokens
                context = context[:, -max_context_length:].contiguous()
            
            logits, _ = model(context)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_decoded = enc.decode(next_token[0].tolist(), errors="ignore")

            yield token_decoded

            # Append the new token to the context for the next step
            context = torch.cat([context, next_token], dim=1).contiguous()

def main():
    # Setup device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Running on device: {device}")

    # Load the model
    model = load_model("./log/model_10B_toks_final.pt")
    model.to(device)

    # Get the tokenizer
    enc = tiktoken.get_encoding("gpt2")

    torch.manual_seed(42)
    prompt = "The"
    # prompt = "There is a"
    print(f"Initial prompt: {prompt}")

    # Instead of building up a massive string,
    # we'll just print tokens as they arrive.
    # That way we don't store enormous text in memory.
    print(prompt, end='', flush=True)
    
    # Adjust max_new_tokens to control total output length
    for token in generate(
        model,
        enc,
        prompt=prompt,
        temperature=0.8,
        top_k=50,
        max_context_length=256,
        max_new_tokens=256,  # e.g. 2000 tokens maximum
    ):
        if token == "<|endoftext|>":
            print("\n")
        else:
            print(token, end='', flush=True)

    print("\nGeneration finished.")

if __name__ == "__main__":
    main()
    