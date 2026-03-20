---
title: "TokTok: A Spanish Tokenizer"
date: "2026-03-08"
description: ""
tags:
  - nlp
  - tokenization
---

So most LLM tokenizers (GPT-4, Llama) were trained on English-heavy data. When they process Spanish, they split words into more pieces than necessary.

For example, GPT-4 sees "aprendizaje" and splits it into `["aprend", "iz", "aje"]`. Three tokens. But if you train a tokenizer specifically on Spanish, it learns that "aprendizaje" is a common word and keeps it as one token.

More tokens = slower inference, more expensive. That's the problem.

## What I built

[TokTok](https://github.com/marcelo-earth/tok-tok) is a BPE tokenizer trained from scratch with SentencePiece on ~100K Spanish Wikipedia articles. I trained three versions: 8K, 32K, and 64K vocab sizes.

The idea is simple: spend the entire vocabulary budget on Spanish instead of splitting it across every language and code.

## How it compares

Ok, here's what happens when you tokenize the same Spanish sentences with different tokenizers:

```
Text: "El aprendizaje automatico permite a las maquinas aprender de los datos."

TokTok 32K (14 tokens): |El|aprendizaje|automa|tico|permite|a|las|ma|quinas|aprender|de|los|datos|.|
GPT-4      (17 tokens): |El|aprend|iz|aje|automatic|o|permite|a|las|ma|qu|inas|aprender|de|los|datos|.|
Llama 3    (18 tokens): |El|aprend|iz|aje|automatic|o|permite|a|las|ma|qu|inas|aprender|de|los|datos|.|
```

GPT-4 breaks "aprendizaje" into three pieces. Llama 3 does the same. TokTok keeps it as two ("aprendizaje" didn't quite make it as one token, but "automa|tico" is still better than "automatic|o").

On a 5MB sample of Spanish Wikipedia, the average compression looks like this:

```
Tokenizer      Vocab     Spanish (chars/token)    English (chars/token)
toktok_8k       8,000     4.71                     2.81
toktok_32k     32,000     5.43                     3.33
toktok_64k     64,000     5.59                     3.68
GPT-4         100,277     4.45                     6.75
Llama 3       128,000     4.34                     6.46
```

toktok_32k gets 5.43 chars per token on Spanish. GPT-4 gets 4.45. That's ~22% better compression with 3x fewer vocabulary entries.

## The tradeoff

But look at the English column. GPT-4 gets 6.75 chars per token on English. toktok_32k gets 3.33. That's terrible for cost. A language-specific tokenizer gains on its target language and loses on everything else, by design.

So when does it make sense?

- Spanish-only workloads: TokTok wins, no question
- Bilingual EN/ES: depends on the split. If it's >70% Spanish, probably still worth it
- Multilingual: stick with GPT-4 or Llama 3

## 32K is the sweet spot

Going from 8K to 32K is a big jump in compression. Going from 32K to 64K, not so much. Diminishing returns.

This makes sense. At 8K you don't have enough vocabulary to capture common words. At 32K you've got most of them. At 64K you're just adding rare words that barely show up.

## Zipf's law

One way to check if a tokenizer has a healthy vocabulary: token frequencies should follow Zipf's law. The most common token appears roughly twice as often as the second most common, three times as often as the third, etc.

TokTok follows this nicely. The frequency distribution on Spanish Wikipedia forms a clean straight line on a log-log plot. That means the vocabulary is well-distributed, no wasted entries on tokens nobody uses.

## What I learned

The comparison is not apples-to-apples, and that's the point. TokTok spends 32K vocab entries on one language. GPT-4 spreads 100K across all languages plus code. Llama 3 spreads 128K.

But per vocabulary entry, TokTok is way more efficient on Spanish. If you're building something that only needs to handle Spanish text, there's no reason to pay the cost of a general-purpose tokenizer.

The tradeoff is real though. You can't have a small, efficient, language-specific tokenizer that also works well on everything else. That's the fundamental tension. And honestly, for most use cases, the general-purpose tokenizers are fine. TokTok is for when "fine" isn't good enough.
