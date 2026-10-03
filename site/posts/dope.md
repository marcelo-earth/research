---
title: "Dope: Comparing Position Encodings"
date: "2026-10-03"
description: "Sinusoidal, learned, RoPE and ALiBi implemented from scratch and compared on perplexity and length generalization"
tags:
  - llm
  - attention
---

**Status: in progress.** All four methods are implemented and unit tested, with training and length generalization scripts. The numeric results from earlier runs were not saved, so this post covers method and qualitative findings. Tables will be added once the runs are saved.

## Question

Self-attention is permutation-invariant, so a transformer needs position information from somewhere. Four common methods are compared on the same model and data: does the choice change perplexity, and does it change behavior on sequences longer than those seen in training?

## Methods

| Method | Modifies | Relative position | Length limit | Origin |
|---|---|---|---|---|
| Sinusoidal | Embeddings (fixed) | Indirect | None | Attention Is All You Need |
| Learned | Embeddings (trained) | No | Fixed by max length | GPT-2 |
| RoPE | Q and K rotation | Yes | None | RoFormer |
| ALiBi | Attention bias | Yes | None | BLOOM |

## Setup

The same base model is used for every run, and only the position encoding changes:

- 6 layers, 8 heads, model dimension 256, about 25M parameters.
- Pre-norm, weight tying between embedding and output head, no bias in linear layers, GELU in the feed-forward block.
- GPT-2 style initialization and GPT-3 style AdamW hyperparameters.
- Data: WikiText-103, training sequence length 256.
- Length generalization: models trained at 256 tokens are evaluated at lengths up to 2048.

## Findings so far

- At this scale and training length, all four methods reach similar perplexity.
- The difference appears in length generalization: training at 256 and testing at 1024 and beyond.
- RoPE and ALiBi handle unseen lengths much better than sinusoidal or learned encodings.
- Learned position embeddings fail completely beyond their maximum length, since no embedding exists for those positions.
- ALiBi adds no parameters. It applies a linear penalty to attention scores based on distance.

## Pending

- Record perplexity per method and the degradation ratio at each test length.
- Add the comparison plots (position similarity matrices, ALiBi slopes, degradation by length).
- Scale the comparison to a larger model and longer evaluation context.
