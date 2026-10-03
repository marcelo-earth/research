---
title: "FlashDance: Flash Attention vs Vanilla Attention"
date: "2026-10-03"
description: "How much faster fused attention really is"
tags:
  - llm
  - attention
---

**Status: in progress.** The benchmark code is complete. The numeric results have not been recorded yet, so this post covers scope, method and qualitative findings. Tables will be added once the runs are saved.

## Question

Flash Attention computes attention without materializing the full N x N score matrix in high-bandwidth memory. The question is how large the gain is in practice, and at what sequence length it starts to matter.

## Setup

Two implementations are compared on the same inputs:

- **Vanilla:** explicit `softmax(QK^T / sqrt(d)) V` with a causal mask.
- **SDPA:** PyTorch `scaled_dot_product_attention`, which dispatches to a fused kernel (Flash Attention on CUDA) when the inputs allow it.

Default configuration: batch size 4, 8 heads, head dimension 64. Timing uses the median over repeated runs, with explicit device synchronization on MPS.

## What is measured

| Dimension | Range |
|---|---|
| Forward pass time | sequence length 128 to 4096 |
| Backward pass time | same range |
| Peak memory | same range |
| Head dimension | 32, 64, 128 |
| Batch size | 1 to 16 |
| Precision | float32, float16, bfloat16 |
| Kernel breakdown | Torch profiler |

## Findings so far

- The speedup of SDPA over vanilla attention grows with sequence length.
- Below roughly 256 tokens the difference is small.
- From about 2K tokens, SDPA uses noticeably less time and memory.
- The backward pass speedup is often larger than the forward pass speedup.
- On CUDA, the Flash Attention kernel is selected automatically through SDPA.
- float16 and bfloat16 add further speedup with minimal precision loss.

## Extended scope

The repository later grew past the original comparison. It now includes implementations of RoPE, ALiBi, grouped query attention (GQA), multi-query attention (MQA), multi-head latent attention (MLA), sliding window attention, cross-attention and a KV cache, each with tests.

It also contains analyses of IO-bound behavior, memory use, prefill vs decode, latency percentiles (p50 to p999), attention entropy and attention sinks, head specialization, and long-context scaling with NTK and position interpolation. A small decoder combines GQA, RoPE, RMSNorm and SwiGLU.

## Pending

- Record forward and backward results with hardware specified.
- Add plots for speedup and memory vs sequence length.
- Repeat on a CUDA GPU, where the Flash Attention kernel is available, to compare against Apple Silicon.

Code: [marcelo-earth/flashdance](https://github.com/marcelo-earth/flashdance)
