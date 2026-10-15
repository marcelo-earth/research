---
title: "BenchPress: A Spanish Benchmark From Scratch"
date: "2025-08-27"
description: "Prompt format moves scores more than you expect, so I am building a Spanish benchmark and checking its labels"
tags:
  - llm
  - evaluation
---

**Status: in progress.** The 500 question set is selected and its key is validated by two blind labelers. Model results on the full set are not in yet, and the set has not been published.

## Question

Running a language model on a benchmark looks like a single number. In practice that number depends on the prompt, the number of examples and the metric. I wanted to see how much, and then build a Spanish benchmark whose answers I can trust.

## What the prompt does

Using lm-eval-harness on small models, changing only the prompt format moved scores by 20 to 40 points. A chain-of-thought prefix sometimes helped and sometimes hurt. Few-shot examples helped, but by very different amounts per task. Asking the same questions in English or Spanish changed the results too. GPT-2 sat near random chance on Spanish questions.

So any comparison between models is only as good as the template, and that has to be fixed and reported with the result.

## Is the key right?

A benchmark with wrong answers measures noise. Before scaling, I set a rule: a pilot of 50 Spanish multiple choice questions needs at least 90% agreement between a blind labeler and the answer key, otherwise I fix the writing process first.

- **Pilot (50 items):** a blind labeler matched the key on 50 of 50, kappa 1.0. The items were easy, so this proved little.
- **Hard set (50 items):** multi step reasoning, orthography and grammar traps, less common facts. Two independent blind labelers matched the key on 50 of 50 and each other on 50 of 50. The key is sound, but the set is **not hard for Claude**.

Both labelers and I belong to the same model family, so this measures agreement with the key, not agreement between humans. I will spot check at least 15 items myself before publishing.

## Do the items separate models?

Local models on the pilot and hard sets, 50 items each, random is 25%:

| Model | Pilot | Hard |
|---|---|---|
| pythia-70m | 26% | 28% |
| gpt2 | 22% | 26% |
| Qwen2.5-0.5B | 42% | 30% |

With 50 items each interval is about 25 points wide. The two small models sit at chance on both sets. Qwen drops from 42% to 30%, which points in the right direction but is not significant (z = 1.26). The sets show the key is sound; they do not yet show that the items discriminate.

## Scaling to 500

I decided the rules before generating anything:

- 600 candidates in 12 topic batches, written with a stated difficulty (30% easy, 40% medium, 30% hard) and a one line justification each.
- Each candidate is answered blind by two independent labelers. An item stays only if both match the key and neither flags it as ambiguous.
- The final 500 are drawn with a fixed seed, stratified by topic and difficulty, without looking at any model result.

Of 599 valid candidates both labelers matched the key on all 599, and 591 survived the ambiguity flags and manual review. The 500 selected items have 152 easy, 200 medium and 148 hard, with the answer letters balanced (A 122, B 127, C 126, D 125).

A labeler agreeing with every key is good for the key and bad as evidence of difficulty. The real test is whether larger open models score above smaller ones. That run comes next, on five local models, and a set where the order by size does not appear will be reported as not discriminating.

## Limitations

- Labelers and authors share a model family. Human agreement is untested until the 15 item spot check.
- The pilot results use 50 items, so the intervals are wide.
- No result yet on the full 500 items.

The code is in [BenchPress](https://github.com/marcelo-earth/benchpress).
