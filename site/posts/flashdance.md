---
title: "FlashDance: Flash Attention vs Vanilla Attention"
date: "2026-10-03"
description: "How much faster fused attention really is"
tags:
  - llm
  - attention
---

**Status: in progress.** The first hardware type is done: on an Apple M3, PyTorch's fused attention is about 3 to 4% slower than the naive version, not faster. CUDA and memory are still pending.

## Question

Flash Attention computes attention without materializing the full N x N score matrix in high-bandwidth memory. The question is how large the gain is in practice, and at what sequence length it starts to matter.

## Setup

Two implementations are compared on the same inputs:

- **Vanilla:** explicit `softmax(QK^T / sqrt(d)) V` with a causal mask.
- **SDPA:** PyTorch `scaled_dot_product_attention`, which dispatches to a fused kernel (Flash Attention on CUDA) when the inputs allow it.

Default configuration: batch size 4, 8 heads, head dimension 64. Timing uses the median over repeated runs, with explicit device synchronization on MPS.

## Planned measurements

| Dimension | Range |
|---|---|
| Forward pass time | sequence length 128 to 4096 |
| Backward pass time | same range |
| Peak memory | same range |
| Head dimension | 32, 64, 128 |
| Batch size | 1 to 16 |
| Precision | float32, float16, bfloat16 |
| Kernel breakdown | Torch profiler |

## Result on Apple Silicon

I fixed the rule before running: a length counts as "SDPA slower" only if the upper bound of the 95% confidence interval is below 1.00, and the smallest effect I care about is 5%. Each block is a fresh process with the order of conditions shuffled from a recorded seed, so every block gives one paired ratio per length. Hardware: MacBook M3, 16 GB, MPS, float32, causal, batch 4, 8 heads, head dimension 64, torch 2.12.

The final run has 25 blocks. Speedup is vanilla time divided by SDPA time, so below 1.00 means SDPA is slower.

| Sequence length | Speedup | 95% CI | Verdict |
|---|---|---|---|
| 512 | 0.967x | [0.954, 0.980] | SDPA slower |
| 1024 | 0.959x | [0.952, 0.967] | SDPA slower |
| 2048 | 0.963x | [0.956, 0.970] | SDPA slower |

A Wilcoxon signed-rank test agrees at all three lengths. The effect is real but small: 3 to 4%, under the 5% I set as relevant. On this hardware, SDPA is not a speedup.

## What went wrong first

The first confirmatory run used 20 blocks and gave 0.974x, 0.984x and 0.939x, with only 2048 resolved. Other processes kept the machine at a load of 5 to 7, and that noise was larger than the effect. I reran with a load gate, which brought the standard deviation of the log ratio down to 0.032, 0.019 and 0.018. The gate is soft, so some blocks still ran under load, but the load of a block did not explain its ratio. Timings drift down over a run on both methods, and the paired ratio absorbs that.

A 6-block pilot underestimated the noise, so the power plan built on it was too optimistic. On a laptop, record the load before trusting a pilot.

## What this does not say

- Memory is not measured yet. The benchmark reports the input tensor size for both methods, so any memory claim waits for a real MPS measurement.
- Only the forward pass, only float32 and only 512 to 2048 tokens are covered here. The backward pass, float16, bfloat16 and the other head dimensions have no recorded numbers.
- The claim "fused attention is faster from about 2K tokens" that this page and the repository README made earlier does not hold on Apple Silicon. The Flash Attention kernel exists on CUDA, not on MPS, so MPS is the control, not the test.

## Extended scope

The repository later grew past the original comparison. It now includes implementations of RoPE, ALiBi, grouped query attention (GQA), multi-query attention (MQA), multi-head latent attention (MLA), sliding window attention, cross-attention and a KV cache, each with tests.

It also contains analyses of IO-bound behavior, memory use, prefill vs decode, latency percentiles (p50 to p999), attention entropy and attention sinks, head specialization, and long-context scaling with NTK and position interpolation. A small decoder combines GQA, RoPE, RMSNorm and SwiGLU.

## Pending

- Run the same forward benchmark on a CUDA GPU on Modal, where SDPA can dispatch to the Flash Attention kernel. That run decides whether the contrast "when Flash matters" holds.
- Measure memory on MPS with `torch.mps.driver_allocated_memory` before saying anything about it.
- Record backward pass, precision and head dimension results with the same design.
- Add plots for speedup and memory against sequence length.

Code: [marcelo-earth/flashdance](https://github.com/marcelo-earth/flashdance)
