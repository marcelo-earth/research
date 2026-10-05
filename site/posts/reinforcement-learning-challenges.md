---
title: "Reinforcement Learning Challenges: From CartPole to GRPO"
date: "2026-10-05"
description: "Reinforcement learning from scratch, up to RL on a small language model"
tags:
  - llm
  - reinforcement-learning
---

**Status: planned.** The repository is set up with the goal and learning path. No exercise has been run yet. This page will be updated as each stage is completed.

## Question

Can a small language model gain 10+ points on GSM8K with GRPO, using an implementation written from scratch instead of a library?

## Path

Each stage has to work before the next one starts:

| # | Stage | Done when |
|---|---|---|
| 1 | Policy gradients (REINFORCE) | CartPole solved |
| 2 | PPO from scratch | Standard control tasks solved reliably |
| 3 | GRPO from scratch | Learning on a verifiable reward |
| 4 | GRPO on a small LLM | 10+ point gain on GSM8K |

## Decision rule

If PPO from scratch does not solve CartPole reliably, GRPO waits. An unstable PPO is the most likely sign that a GRPO run on a language model would be unstable or too costly too.

## Next step

Solve CartPole with a from-scratch policy gradient.

Code: [marcelo-earth/reinforcement-learning-challenges](https://github.com/marcelo-earth/reinforcement-learning-challenges)
