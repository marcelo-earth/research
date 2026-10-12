---
title: "ARC Transformer"
date: "2026-10-11"
description: "Reproducing a small transformer trained from scratch on ARC puzzles"
tags:
  - llm
  - arc
  - reproduction
---

**Status: partial.** The point reproduction is done and verified. The ablations and the under-$1 cost target are not.

## Question

mdlARC, a blog post and repo by mvakde, reports that a 75M parameter decoder-only transformer trained from scratch reaches 44% on the ARC-AGI-1 public evaluation set. The author says it took about 1.5 hours on one RTX 5090 for roughly $0.67.

I wanted to know two things. Can I get the same score with the same recipe? And does the cost hold up on a different GPU?

## Setup

I used the reference code pinned at commit `8afc20d` and changed nothing in the algorithm. The only difference is the hardware: one H100 80 GB on Modal instead of the 5090 on vast.ai.

- Model: 8 layers, 768 dimensions, 12 heads, 3D RoPE and learned embeddings per example and per dihedral transform.
- High preset: 650 epochs, up to 300 augmented views at inference, seed 42, batch size 32.
- Training data includes the demonstration pairs of the evaluation puzzles (test-time training), ConceptARC and the other ARC-1 tasks. The test outputs stay hidden.
- Score: task-averaged pass@2 on all 400 public evaluation tasks, taken from checkpoint 648 only. I wrote that rule down before launching, so I could not pick the best checkpoint after seeing results.

I also set a decision band before the run: anything from 42% to 46% counts as consistent with the reported 44%.

## Result

| Run | Epochs | Score (pass@2) | Fully solved tasks |
|---|---|---|---|
| Low pilot | 90 | 33.875% | 134 |
| High | 650 | **42.75%** | 169 |

The high run scored 171 out of 400 tasks, which is **42.75%**. That is 1.25 points short of the 44% in the reference, inside the band I set. 169 tasks were solved completely. The score is higher than that count would suggest because tasks with several test pairs earn partial credit.

All 400 tasks and all 419 test pairs were covered. I rescored the submission with a separate script and got the same number, the same task IDs and the same fingerprint for the solutions file.

So: 42.75% of 44%. Close, not exact.

## What went wrong along the way

Training finished all 650 epochs in 67 minutes. Then the evaluation stopped at batch 546 of 1008 because Modal reported the input as cancelled. I don't know what sent that signal and I won't guess.

The checkpoints were already saved, so I didn't retrain. I ran only the evaluation again on checkpoint 648, as a spawned call whose result I collected from a separate process, and it finished in about 20 minutes. If I had started a fresh training run instead, I would have paid for the same 67 minutes twice.

## Cost

Tagged Modal compute for the whole high run, including the cancelled evaluation and the recovery, was **$7.31**. The first run cost $5.85 and the recovery $1.46. My account credits covered it, so nothing was billed in cash.

That is about ten times the reference's $0.67. The target of under $1 was not met, and a matching score doesn't change that. An H100 is a lot more expensive per hour than a 5090 on vast.ai, and I didn't try to shrink the run.

## Limits

- One training seed. I can't say how much the score moves between seeds, so there is no confidence interval and no claim of statistical equivalence. The two-point band is a tolerance I picked, not a test.
- The jump from 33.9% to 42.75% changes two things at once, the number of training epochs and the number of inference views. It doesn't isolate the effect of more training.
- Different hardware from the reference. Same algorithm, different cost.

## What is next

This post covers the reproduction only. The plan was to ablate 3D RoPE and the per-example embeddings, then try to pass 50% with changes to positional encoding. That needs new compute, and I'll plan it before spending more of the credits. A second seed would tell me how much to trust the 42.75% in the first place, so it probably goes first.

The reference is [mdlARC by mvakde](https://github.com/mvakde/mdlARC). Code, run logs and every run's cost are in the [arc-transformer](https://github.com/marcelo-earth/arc-transformer) repo.
