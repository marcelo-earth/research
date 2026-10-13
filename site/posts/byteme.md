---
title: "ByteMe: Four Tokenizers, One Text"
date: "2026-10-12"
description: "Byte-level BPE vs BPE vs WordPiece vs Unigram, and the unknown tokens that tokenize() hides"
draft: true
tags:
  - llm
  - tokenizers
---

Four tokenizers, the same 13 texts. Byte-level BPE never loses a character. T5's Unigram tokenizer looks like the best at Japanese and Arabic until you check the ids: most of that text becomes `<unk>`.

## Question

Language models see tokens, not text. There are four common ways to cut text into tokens. How differently do they treat the same input, and what does each one lose?

## Setup

One real pretrained tokenizer per strategy, loaded from Hugging Face:

| Strategy | Tokenizer | Vocab | How it builds tokens |
|---|---|---:|---|
| Byte-level BPE | `gpt2` | 50,257 | Starts from the 256 bytes, merges frequent pairs |
| BPE | `EleutherAI/pythia-70m` | 50,254 | Also byte-level BPE in practice, trained on The Pile instead of WebText |
| WordPiece | `bert-base-uncased` | 30,522 | Merges by likelihood, lowercases, `##` marks a word continuation |
| Unigram | `t5-small` | 32,100 | Starts from a large vocab and prunes the pieces that hurt the least |

Thirteen texts: English, Spanish, Python code, emoji, URLs, mixed content, rare words, numbers, made-up words, Japanese, Arabic, mixed scripts and a keyboard smash. Every count below comes from `encode()` without special tokens, so an unknown character shows up as the unknown token id.

## Result

Token count per text. Fewer is better, as long as nothing is thrown away.

| Text | Chars | Byte-level BPE | BPE | WordPiece | Unigram |
|---|---:|---:|---:|---:|---:|
| English | 120 | 21 | 21 | 21 | 23 |
| Spanish | 155 | 47 | 38 | 45 | 57 |
| Code | 136 | 67 | 56 | 55 | 59 (1 unk) |
| Emoji | 72 | 32 | 35 | 17 (3 unk) | 19 (3 unk) |
| URLs | 144 | 46 | 43 | 49 | 50 |
| Mixed | 94 | 35 | 35 | 36 (1 unk) | 39 (2 unk) |
| Rare words | 111 | 29 | 30 | 32 | 35 |
| Numbers | 101 | 28 | 28 | 29 | 25 |
| Made-up words | 67 | 17 | 17 | 20 | 24 |
| Japanese | 16 | 31 | 19 | 16 (9 unk) | 2 (1 unk) |
| Arabic | 36 | 35 | 26 | 31 | 12 (6 unk) |
| Mixed scripts | 27 | 26 | 19 | 19 (2 unk) | 12 (4 unk) |
| Keyboard smash | 36 | 23 | 23 | 26 | 31 (1 unk) |

What stands out:

- **On English, the strategy barely matters.** Three of the four give 21 tokens for the same sentence.
- **GPT-2 and Pythia never produce an unknown token.** Anything outside the vocab falls back to bytes. The price is length: 16 Japanese characters become 31 GPT-2 tokens, about two tokens per character.
- **Pythia beats GPT-2 on Spanish, code, Japanese and Arabic** with the same algorithm and almost the same vocab size. The only real difference is the training data.
- **WordPiece drops emoji and most Japanese** to `[UNK]`: 9 of 16 Japanese characters are gone.
- **Unigram's best numbers are its worst results.** Japanese in 2 tokens and Arabic in 12 look like great compression, but one of those 2 tokens and 6 of those 12 are `<unk>`. The whole Japanese sentence becomes a single unknown token.

## What went wrong first

The first version of this comparison said Unigram was "surprisingly good at multilingual". That came from `tokenize()`, which returns the pieces the pre-tokenizer cuts before they are looked up in the vocab. For T5, `tokenize("こんにちは")` returns `["▁", "こんにちは"]`, which looks fine. `encode()` returns `[3, 2]`, and id 2 is `<unk>`. T5's vocab was built for English, French, German and Romanian, so most non-Latin text has nowhere to go.

The interactive page had the same bug: it showed 0 unknown tokens for Unigram on Arabic. It now maps text to ids first and back to tokens, and the README and notebook takeaway are corrected.

A second bug came from the browser. Transformers.js loads the `gpt2` repo's `tokenizer.json` without its model type, so it skipped the BPE merges and returned whole words: `Supercalifragilisticexpialidocious` came out as 1 token instead of 11. The `Xenova/gpt2` mirror fixes it.

## Interactive version

The repository includes a static page: type any text and see the four tokenizers side by side, with colored tokens, token count, characters per token and unknown tokens highlighted. Tokenization runs in the browser with Transformers.js, with no backend. On 8 preset texts, all 32 tokenizer and text pairs give the same tokens in the browser as in Python.

## What this does not say

- The "BPE" column is not a separate algorithm. Pythia's GPT-NeoX tokenizer is byte-level too, so this is really two byte-level BPE tokenizers trained on different data. A character-level BPE without byte fallback would show unknown tokens.
- These are four specific tokenizers, not the four strategies in general. A Unigram tokenizer trained on multilingual data, like mT5's or XLM-R's, would not drop Japanese.
- GPT-2 stands in for byte-level BPE. GPT-4 and Llama 3 use the same strategy with much larger vocabs, so their counts would be lower.
- Thirteen texts is a demonstration, not a benchmark. There is no corpus-level compression number here.

## Pending

- Host the interactive page.
- Add a multilingual Unigram tokenizer (mT5) next to T5 to separate the algorithm from the vocab.

Code: [marcelo-earth/byte-me](https://github.com/marcelo-earth/byte-me)
