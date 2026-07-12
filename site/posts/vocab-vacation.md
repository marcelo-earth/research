---
title: "VocabVacation: Does Vocab Size Matter?"
date: "2026-03-30"
description: "Training tiny transformers with different vocab sizes to find the sweet spot"
tags:
  - llm
  - tokenization
---

So bigger vocab means each token carries more information. GPT-4 has 100K tokens, Llama 3 has 128K. But there's a cost: the embedding table.

Every token gets a vector. If your vocab is 32K and your embedding dimension is 256, that's 8M parameters just to store the lookup table. For a 7B model, that's nothing. For a 15M model, that's more than half the entire network.

I wanted to see where it breaks.

## The experiment

I trained four GPT-style transformers on WikiText-103, all with roughly the same total parameter count (~14M). The only thing that changed was the vocab size: 1K, 4K, 8K, 32K.

The trick is scaling model dimensions to keep total params constant. Bigger vocab means smaller dimensions:

```python
configs = {
    1000:  {"dim": 384, "n_heads": 6, "n_layers": 6},   # ~11M params
    4000:  {"dim": 320, "n_heads": 8, "n_layers": 6},    # ~13M params
    8000:  {"dim": 288, "n_heads": 6, "n_layers": 6},    # ~14M params
    32000: {"dim": 192, "n_heads": 6, "n_layers": 6},    # ~14M params
}
```

Why scale down the dimension? Because if you don't, the 32K model would have way more parameters than the 1K model, and the comparison wouldn't be fair.

## Where do the parameters go?

Before training anything, look at this:

```
Vocab    Dim    Embedding %    Transformer %
1K       384       9%              91%
4K       320      19%              81%
8K       288      29%              71%
32K      192      53%              47%
```

At 32K vocab, more than half the model is just the embedding table. That's 53% of your parameter budget spent on a lookup table, leaving only 47% for actual self-attention and feed-forward layers.

At 1K, it's the opposite: 91% of the model is transformer layers. Lots of capacity for language understanding, but each token carries very little information.

## Results

```
Vocab    Dim    Total params    Emb %    Compression    Perplexity
1K       384       ~11M          9%       2.31            high
4K       320       ~13M         19%       3.72            medium
8K       288       ~14M         29%       4.58            lowest
32K      192       ~14M         53%       5.89            medium-high
```

8K wins. Perplexity drops fast from 1K to 8K, then goes back up at 32K.

Why does 32K do worse than 8K? Because the model doesn't have enough transformer capacity left. The embedding table ate the budget.

## The sequence length problem

There's another issue with small vocabs. The same text becomes way more tokens:

```
Text: "The quick brown fox jumps over the lazy dog."

Vocab 1K:   22 tokens (2.0 chars/tok)
Vocab 4K:   12 tokens (3.7 chars/tok)
Vocab 8K:   10 tokens (4.4 chars/tok)
Vocab 32K:   8 tokens (5.5 chars/tok)
```

1K vocab splits everything into tiny pieces. More tokens means longer sequences, slower training, more memory. The attention mechanism is O(n²) with respect to sequence length, so doubling the number of tokens makes attention 4x more expensive.

## Why real LLMs use 100K+ vocab

At 7B parameters, a 128K vocab with dimension 4096 is about 500M parameters. That's ~7% of the model. The embedding table is a rounding error.

But the compression benefits are huge. Shorter sequences mean:
- Faster inference (fewer forward passes for autoregressive generation)
- Cheaper attention (O(n²) with fewer tokens)
- Longer effective context (same window fits more text)

For small models, the math flips. The embedding table dominates, and you don't have enough capacity left for the transformer to learn anything useful.

## The sweet spot depends on model size

This is the key insight. There's no universal "best vocab size." It depends on how many parameters you have.

- **Under 25M params**: 4K-8K vocab
- **25M-100M params**: 8K-32K vocab
- **1B+ params**: 32K-128K vocab is fine
- **7B+ params**: Go as high as you want

The formula is simple: if your embedding table is more than ~30% of total params, your vocab is too big for your model.

That's why [Vowel](https://github.com/marcelo-earth/vowel) exists. The code is there if you want to run it yourself.
