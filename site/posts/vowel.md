---
title: "Vowel: Measuring Vocab Size Honestly"
date: "2025-08-20"
description: "A vocab size sweep, and the metric bug that made it incomparable"
tags:
  - llm
  - tokenization
---

**Status: in progress.** The first sweep is done and its metric turned out to be flawed. The fixed 5 by 3 sweep started and has only one finished run so far, so there is no final curve yet.

## Question

At a fixed parameter budget, how does vocabulary size change how well a small language model compresses text? A bigger vocab shortens sequences but moves parameters from the transformer into the embedding table.

## Setup

Same corpus (the first 50,000 lines of WikiText-103), same depth, about 15M parameters per run. The width is solved per vocab size so the total stays constant. Three epochs, learning rate 3e-4.

| Vocab | Dim | Params | Embedding share |
|---|---|---|---|
| 1K | 450 | 15.2M | 3% |
| 4K | 426 | 14.9M | 11% |
| 8K | 402 | 15.0M | 22% |
| 32K | 282 | 14.8M | 61% |

## First sweep

One seed per size, validation bits per character after epoch 3:

| Vocab | Bits per char |
|---|---|
| 1K | 1.799 |
| 4K | 1.850 |
| 8K | 1.806 |
| 32K | 1.753 |

Only 32K stands out, and with a single seed there is no interval to say whether even that is real. The gaps between 1K, 4K and 8K are at most 0.05 bits, which is the kind of difference seed noise can produce.

## The metric was wrong

Bits per character divides the loss by the number of characters in the validation text. Two things broke that comparison:

- The tokenizer used a whitespace pre-tokenizer, which **drops spaces**. The model never had to predict them, but the denominator still counted them.
- Validation used windows that dropped the last partial batch, while the denominator covered the full text.

Both errors depend on the vocab size, so they bias the curve instead of adding uniform noise. The fix was to use a byte-level BPE tokenizer, which is lossless, and to count bytes exactly from the tokens that were actually scored. Bits per char is then bits per byte times a constant. The tests check that the tokenizer round-trips text and that the byte count matches. The first sweep is superseded and not comparable with anything measured after the fix.

## The plan before running

The second sweep is registered before any run: vocab sizes 1K, 4K, 8K, 16K and 32K, seeds 42, 43 and 44, 15 runs in total.

- The unit of replication is the seed. Each vocab gets a mean and a 95% t interval with n = 3.
- Two sizes count as different only if the Welch interval of the difference excludes zero.
- With 3 runs against 3 runs, the smallest difference I can detect with 80% power is about 3.1 seed standard deviations. Anything smaller gets reported as not resolved, never as a trend.
- If the seed spread is larger than the gap between neighboring sizes, that is the result, and I stop.

## What happened so far

The first run of the new sweep (1K, seed 42) finished at 1.749 bits per byte, 1.752 bits per char. Its first epoch took over two hours because the machine was busy. After that the sweep process crashed with a Metal shader compiler error, so 14 of 15 runs are still pending. The sweep resumes from where it stopped, so it needs a restart and a clear machine, not a rewrite.

## Limitations

- One run is not a curve. Nothing in this post should be read as a trend across vocab sizes.
- Fifteen million parameters and three epochs is far from where vocab size decisions matter for real models.
- Everything runs on a laptop GPU, which makes wall-clock time noisy and slow.

The code is in [Vowel](https://github.com/marcelo-earth/vowel). For the parameter accounting behind this, see [VocabVacation](/posts/vocab-vacation/).
