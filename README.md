# GPT-2 Optimisation and Training

This code primarily revolves around optimising the training process of a GPT-2 model, drawing inspiration from resources like nanoGPT and implementing techniques used in large-scale training like GPT-3. The goal is to maximise training efficiency and performance so that a model that took around a week or so to train in 2019 can be trained in a few hours.

The project explores various optimisation strategies documented within the `gpt-2-optimisation.ipynb` notebook. A separate notebook, `gpt-2-from-scratch.ipynb`, details the initial implementation based on the GPT-2 small architecture.

## Key Features & Optimisations Implemented

- **Mixed Precision:** Utilises `torch.set_float32_matmul_precision('high')` for TF32 on compatible hardware and `torch.autocast` (with BFloat16 or Float16) for model forward pass and loss calculations to reduce memory usage and leverage Tensor Cores.
- **Flash Attention:** Replaces the standard self-attention mechanism with `nn.functional.scaled_dot_product_attention` for improved speed and memory efficiency. The fused kernel is cracked (will learn implementation one day)
![Scaled Dot Product Attention](/notebooks//images/flash-attention.png)
- **Vocabulary Size Padding:** The vocabulary size is padded to the nearest multiple of 64 (50304) to improve computational alignment on GPUs and allow for future special token support.
- **Torch Compile:** Integrates `torch.compile()` to potentially fuse kernels and reduce Python overhead.
- **Optimised AdamW:**
  - Uses hyperparameters closer to GPT-3 (`betas=(0.9, 0.95)`, `eps=1e-8`).
  - Selectively applies weight decay (0.1) only to multi-dimensional parameters (weights), excluding masks and LayerNorm parameters.
  - Utilises the `fused` AdamW implementation when on CUDA.
- **Gradient Clipping:** Applies global gradient norm clipping with a maximum norm of 1.0 (`torch.nn.utils.clip_grad_norm_`) to stabilise training.
- **Learning Rate Scheduling:** Implements a cosine decay learning rate schedule with linear warmup, following GPT-3 practices.
![Learning Rate over Training](/notebooks/images/cosine-decay-lr.png)
- **Distributed Data Parallel (DDP):** Sets up distributed training and wraps the model with `DistributedDataParallel` for multi-GPU training, synchronising gradients efficiently. 
- **Gradient Accumulation:** Simulates larger batch sizes by accumulating gradients over multiple smaller batches before performing an optimiser step, targeting an effective batch size of ~0.5M tokens. This matches the original batch size of the GPT2 model. Gather operations are manually completed after all gradient accumulation is complete to minimise cross GPU communication.

## Training Details

- **Model:** GPT-2-small architecture (specifics defined in `GPTConfig`).
- **Dataset:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), specifically the 10 billion token subset (`edu_fineweb10B`). Education data had been shown to be very effective for quickly getting high performance in benchmarks such as MMLU.
- **Training:** The model was configured to train for 1 epoch over the 10B token dataset.
