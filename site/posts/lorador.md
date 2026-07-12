---
title: "LoRAdor: LoRA from Scratch"
date: "2026-03-30"
description: "Implementing Low-Rank Adaptation in pure PyTorch"
tags:
  - llm
  - fine-tuning
---

Fine-tuning a 7B model means updating 7 billion parameters. That requires a lot of memory, a lot of compute, and a full copy of the model for every task. LoRA changes that.

The idea is simple: instead of updating the full weight matrix, you add a small low-rank decomposition on top of it. The original weights stay frozen. You only train two tiny matrices.

## The math

A linear layer does `y = Wx`. During fine-tuning, the weight changes: `W' = W + ΔW`.

LoRA says: instead of storing the full ΔW (which is huge), decompose it into two small matrices:

$$
\Delta W = B \times A
$$

Where:
- W is `d_out × d_in` (e.g., 4096 × 4096 = 16M parameters)
- A is `rank × d_in` (e.g., 8 × 4096 = 32K parameters)
- B is `d_out × rank` (e.g., 4096 × 8 = 32K parameters)

That's 64K trainable parameters instead of 16M. A 250x reduction for a single layer.

## Implementation

Here's LoRA in pure PyTorch:

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad_(False)

        d_in = original_layer.in_features
        d_out = original_layer.out_features

        # A: initialized with random normal (so the product starts non-zero)
        self.A = nn.Parameter(torch.randn(d_in, rank) / rank)
        # B: initialized with zeros (so ΔW starts at zero)
        self.B = nn.Parameter(torch.zeros(rank, d_out))

        self.scale = alpha / rank

    def forward(self, x):
        # original path (frozen) + low-rank path (trainable)
        return self.original(x) + (x @ self.A @ self.B) * self.scale
```

That's it. Three key decisions:

1. **B starts at zero**. This means `ΔW = B × A = 0` at initialization. The model starts exactly where the pre-trained model left off. No random noise at the start.

2. **A starts with small random values**. If both were zero, gradients would be zero and nothing would ever update.

3. **The scale factor `alpha/rank`**. This controls how much influence the LoRA path has. `alpha` is a hyperparameter, typically 2x the rank.

## Applying it to a transformer

You don't LoRA every layer. Typically, you target the attention projections:

```python
def add_lora(model, rank=8, alpha=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            key in name for key in ["q_proj", "k_proj", "v_proj", "o_proj"]
        ):
            parent_name = name.rsplit(".", 1)[0]
            attr_name = name.rsplit(".", 1)[1]
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, attr_name, LoRALayer(module, rank, alpha))
    return model
```

Now you can freeze the base model and only train the LoRA parameters:

```python
# freeze everything
for param in model.parameters():
    param.requires_grad_(False)

# add LoRA (only A and B are trainable)
model = add_lora(model, rank=8, alpha=16)

# check: how many params are trainable?
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}  Trainable: {trainable:,}  ({100*trainable/total:.2f}%)")
```

For a 7B model with rank 8, you'll see something like 0.1% of parameters are trainable. That's the whole point.

## Why rank 8 is enough

This is the surprising part. A 4096×4096 matrix has 16M degrees of freedom, but the *change* you need for fine-tuning lives in a much lower-dimensional space.

Think about it: fine-tuning for a specific task doesn't fundamentally rewire the model. It nudges the existing representations slightly. That nudge can be captured by a rank-8 matrix.

The LoRA paper tested ranks from 1 to 64:
- Rank 1: works for simple tasks
- Rank 4-8: works for most fine-tuning
- Rank 64: diminishing returns, barely better than rank 8
- Full rank: no better than rank 64

## The cost comparison

For Llama 3.1 8B with QLoRA (4-bit quantization + LoRA):

```
Method              VRAM      Trainable params
Full fine-tune      ~60 GB    8B (100%)
LoRA rank 16        ~18 GB    ~20M (0.25%)
QLoRA rank 16       ~6 GB     ~20M (0.25%)
QLoRA rank 32       ~7 GB     ~40M (0.5%)
```

QLoRA on a single T4 (16GB) can fine-tune an 8B model. Full fine-tuning needs 4x A100s. Same model, same task, similar results.

## When to use what rank

- **Rank 4**: Quick experiments, simple tasks (sentiment, classification)
- **Rank 8-16**: General fine-tuning, instruction following
- **Rank 32**: Complex tasks where you need more capacity (code, math)
- **Rank 64+**: Probably overkill, but try it if lower ranks underperform

The `alpha` parameter is usually set to 2× the rank. So rank 8 gets alpha 16. This is a convention, not a law, but it works well in practice.

## What I use in practice

In [Generative Manim](https://github.com/marcelo-earth/generative-manim), I use QLoRA with rank 32 and alpha 64 across a three-stage pipeline: SFT, DPO, then GRPO. All on Kaggle T4s. Total cost: around $30.

The key config:

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",
    lora_dropout=0.05,
)
```

`target_modules="all-linear"` applies LoRA to every linear layer, not just attention. This uses more memory but captures more of the fine-tuning signal. For rank 32, it's worth it.

## The intuition

LoRA works because fine-tuning is a low-rank operation. You're not teaching the model a new language. You're steering it toward a specific behavior. That steering happens in a low-dimensional subspace of the full weight space.

The frozen weights hold the knowledge. The LoRA matrices hold the direction.
