---
title: "UltraLlama: Serving Benchmark on vLLM"
date: "2026-10-05"
description: "How much throughput tuning vLLM settings really buys"
tags:
  - llm
  - inference
---

**Status: planned.** The repository has the scaffold: benchmark plan, a config for the run matrix and a placeholder load runner. No server has been launched and no numbers exist yet. This page will be updated as runs are recorded.

## Question

Default serving settings are rarely the best ones for a given workload. The question is how much throughput a tuned vLLM configuration gains over the defaults for one realistic workload, at the same latency target.

## Setup

- **Server:** vLLM serving `Llama-3.1-8B-Instruct`.
- **Load:** single-user and concurrent traffic, with 1, 4 and 8 users and 20 to 80 requests per profile.
- **Comparison:** default settings against tuned ones, sweeping the main serving knobs.

## What will be measured

| Metric | Why |
|---|---|
| Time to first token (TTFT) | What a user feels before output starts |
| Output tokens per second | Raw throughput |
| p50 and p95 latency | Behavior under load, not just the average |
| Error rate | Where the server stops coping |

## Decision rule

The target is a 2x throughput gain at equal latency. If tuning yields under 1.3x after the main knobs are swept, the result becomes a sensitivity analysis of which settings matter and which do not.

## Next step

Start the vLLM baseline server and log one single-user latency run.

Code: [marcelo-earth/ultrallama](https://github.com/marcelo-earth/ultrallama)
