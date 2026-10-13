# Recovered token usage of the main run

The main run kept only each answer's text. This table set restores per-question token usage from
the providers' own records (Neuralwatt's usage API, LLM Gateway's request logs), which carry no
question text and were tied to questions after the fact; see [README](../../README.md#how-the-usage-was-recovered)
for the method. Counts are each provider's, in each model's own tokenizer: compare languages within
a model, not counts across models.

- Reasoning tokens are **reported** by LLM Gateway, and **estimated** for the Neuralwatt models as
  output tokens − answer tokens − a fixed overhead (2; Gemma 5), which matched the reported or
  tokenized reasoning within ±1 token on re-run requests. A missing gateway value counts as 0.
- Medians and ratios use only matched questions.

## Coverage

Matched: questions tied to a usage record (of 5,880). Check: on questions whose record is known independently (re-run requests, or anchors held out of the alignment), how many were tied to the right record; misses were swaps between neighbouring questions. GPT-6 Luna and GPT-6.1 Sol miss the 68 Swahili math questions blocked by the content filter. Any reasoning: share of matched questions with more than 0 reasoning tokens.

| Model | Matched | Method | Check | Reasoning tokens | Any reasoning |
| --- | --- | --- | --- | --- | --- |
| Claude Haiku 5.5 | 5,825 | anchored order | 147/159 (held-out anchors) | reported | 99% |
| GPT-6 Luna | 5,811 | exact prompt tokens | exact | reported | 100% |
| Gemini 3.8 Flash | 5,804 | anchored order | 103/110 (held-out anchors) | reported | 100% |
| Claude Sonnet 5.5 † | 5,772 | anchored order | 140/159 (held-out anchors) | reported | 14% |
| GPT-6.1 Sol † | 5,812 | exact prompt tokens | exact | reported | 44% |
| DeepSeek V4.1 Flash | 5,879 | exact prompt tokens | 180/180 (re-run requests) | estimated | 100% |
| GLM-5.3 † | 5,879 | exact prompt tokens | 294/294 (re-run requests) | estimated | 100% |
| GLM-5.3-Flash | 5,880 | exact prompt tokens | 341/341 (re-run requests) | estimated | 100% |
| Qwen 3.8 27B | 5,877 | exact prompt tokens | 107/107 (re-run requests) | estimated | 100% |
| Gemma 4 31B | 5,879 | exact prompt tokens | 383/385 (re-run requests) | estimated | 100% |

## Reasoning tokens by language

**Median reasoning tokens per question:**

| Model | en | zh | ja | ko | pt | de | id | ru | ar | sw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Haiku 5.5 | 64 | 61 | 69 | 67 | 71 | 76 | 71 | 68 | 69 | 74 |
| GPT-6 Luna | 106 | 126 | 141 | 134 | 111 | 114 | 121 | 120 | 134 | 134 |
| Gemini 3.8 Flash | 309 | 362 | 386 | 361 | 364 | 384 | 372 | 375 | 392 | 432 |
| Claude Sonnet 5.5 † | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GPT-6.1 Sol † | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 21 |
| DeepSeek V4.1 Flash | 240 | 510 | 450 | 405 | 328 | 430 | 386 | 390 | 430 | 458 |
| GLM-5.3 † | 351 | 381 | 428 | 361 | 342 | 392 | 354 | 340 | 372 | 484 |
| GLM-5.3-Flash | 267 | 318 | 354 | 340 | 272 | 292 | 308 | 265 | 314 | 366 |
| Qwen 3.8 27B | 459 | 590 | 1,120 | 1,013 | 816 | 780 | 612 | 840 | 635 | 1,646 |
| Gemma 4 31B | 920 | 938 | 888 | 872 | 837 | 900 | 828 | 876 | 874 | 912 |

**Relative to English**: total reasoning tokens in the language ÷ total on the same questions in English. Part of a difference can be tokenization (some scripts take more tokens) or the language the model reasons in, not only more reasoning.

| Model | zh | ja | ko | pt | de | id | ru | ar | sw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Haiku 5.5 | 1.05 | 1.08 | 1.05 | 1.12 | 1.24 | 1.10 | 1.11 | 1.19 | 1.18 |
| GPT-6 Luna | 1.01 | 1.25 | 1.09 | 1.06 | 1.02 | 1.07 | 1.09 | 1.08 | 1.21 |
| Gemini 3.8 Flash | 1.26 | 1.26 | 1.16 | 1.15 | 1.24 | 1.23 | 1.14 | 1.27 | 1.31 |
| Claude Sonnet 5.5 † | 1.76 | 2.02 | 2.01 | 1.71 | 1.86 | 1.62 | 1.65 | 2.34 | 2.15 |
| GPT-6.1 Sol † | 1.13 | 1.19 | 1.11 | 1.10 | 1.09 | 1.07 | 1.12 | 1.05 | 1.25 |
| DeepSeek V4.1 Flash | 1.34 | 1.55 | 1.58 | 1.28 | 1.31 | 1.43 | 1.29 | 1.31 | 1.79 |
| GLM-5.3 † | 1.31 | 1.21 | 1.06 | 1.21 | 1.39 | 1.30 | 0.99 | 1.15 | 1.48 |
| GLM-5.3-Flash | 1.21 | 1.35 | 1.14 | 0.87 | 1.12 | 1.16 | 0.94 | 1.11 | 1.16 |
| Qwen 3.8 27B | 1.09 | 1.63 | 1.69 | 1.31 | 1.17 | 1.13 | 1.38 | 1.30 | 1.84 |
| Gemma 4 31B | 1.00 | 0.93 | 0.88 | 0.87 | 0.90 | 0.83 | 0.92 | 0.84 | 0.93 |

## Output tokens by language

All output tokens (reasoning + answer, `completion_tokens`). **Median per question:**

| Model | en | zh | ja | ko | pt | de | id | ru | ar | sw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Haiku 5.5 | 459 | 645 | 786 | 851 | 731 | 874 | 802 | 708 | 935 | 886 |
| GPT-6 Luna | 186 | 208 | 284 | 228 | 209 | 209 | 221 | 212 | 224 | 232 |
| Gemini 3.8 Flash | 474 | 566 | 623 | 583 | 572 | 609 | 586 | 593 | 618 | 672 |
| Claude Sonnet 5.5 † | 196 | 231 | 292 | 317 | 249 | 291 | 291 | 271 | 324 | 348 |
| GPT-6.1 Sol † | 98 | 136 | 189 | 144 | 123 | 131 | 135 | 136 | 120 | 139 |
| DeepSeek V4.1 Flash | 359 | 618 | 618 | 567 | 468 | 582 | 526 | 548 | 556 | 610 |
| GLM-5.3 † | 544 | 623 | 759 | 658 | 622 | 654 | 686 | 604 | 672 | 826 |
| GLM-5.3-Flash | 448 | 519 | 668 | 675 | 504 | 530 | 572 | 495 | 602 | 678 |
| Qwen 3.8 27B | 686 | 801 | 1,442 | 1,254 | 1,080 | 1,032 | 885 | 1,077 | 916 | 2,086 |
| Gemma 4 31B | 1,376 | 1,434 | 1,364 | 1,331 | 1,307 | 1,400 | 1,284 | 1,400 | 1,334 | 1,456 |

**Relative to English** (as above):

| Model | zh | ja | ko | pt | de | id | ru | ar | sw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Haiku 5.5 | 1.35 | 1.50 | 1.58 | 1.44 | 1.74 | 1.59 | 1.44 | 1.81 | 1.68 |
| GPT-6 Luna | 1.02 | 1.39 | 1.12 | 1.09 | 1.06 | 1.10 | 1.10 | 1.10 | 1.21 |
| Gemini 3.8 Flash | 1.23 | 1.27 | 1.18 | 1.16 | 1.22 | 1.22 | 1.16 | 1.27 | 1.32 |
| Claude Sonnet 5.5 † | 1.19 | 1.36 | 1.48 | 1.25 | 1.45 | 1.37 | 1.27 | 1.53 | 1.62 |
| GPT-6.1 Sol † | 1.26 | 1.64 | 1.32 | 1.18 | 1.23 | 1.24 | 1.23 | 1.14 | 1.27 |
| DeepSeek V4.1 Flash | 1.32 | 1.55 | 1.55 | 1.26 | 1.31 | 1.41 | 1.28 | 1.29 | 1.73 |
| GLM-5.3 † | 1.31 | 1.24 | 1.09 | 1.22 | 1.39 | 1.32 | 1.01 | 1.17 | 1.49 |
| GLM-5.3-Flash | 1.20 | 1.38 | 1.18 | 0.90 | 1.14 | 1.18 | 0.96 | 1.13 | 1.18 |
| Qwen 3.8 27B | 1.09 | 1.62 | 1.65 | 1.30 | 1.18 | 1.14 | 1.37 | 1.30 | 1.84 |
| Gemma 4 31B | 1.02 | 0.95 | 0.91 | 0.90 | 0.93 | 0.87 | 0.96 | 0.88 | 0.97 |

## Right and wrong answers

Median reasoning tokens per question, all languages:

| Model | Right | Wrong |
| --- | --- | --- |
| Claude Haiku 5.5 | 66 | 108 |
| GPT-6 Luna | 112 | 302 |
| Gemini 3.8 Flash | 359 | 638 |
| Claude Sonnet 5.5 † | 0 | 0 |
| GPT-6.1 Sol † | 0 | 49 |
| DeepSeek V4.1 Flash | 347 | 1,286 |
| GLM-5.3 † | 317 | 1,912 |
| GLM-5.3-Flash | 267 | 977 |
| Qwen 3.8 27B | 656 | 4,324 |
| Gemma 4 31B | 807 | 1,678 |

## Token cap and empty answers

Hit cap: `finish_reason = length` (the output reached the model's max tokens). Empty answers: as logged by lm-eval, by the finish reason of their record (no record: the blocked Swahili math questions of the GPT models).

| Model | Hit cap | Empty answers | Finish reason of empty answers |
| --- | --- | --- | --- |
| Claude Haiku 5.5 | 0 | 0 | – |
| GPT-6 Luna | 0 | 68 | no record 68 |
| Gemini 3.8 Flash | 0 | 7 | SAFETY 7 |
| Claude Sonnet 5.5 † | 0 | 0 | – |
| GPT-6.1 Sol † | 0 | 68 | no record 68 |
| DeepSeek V4.1 Flash | 4 | 5 | length 4, stop 1 |
| GLM-5.3 † | 97 | 93 | length 92, stop 1 |
| GLM-5.3-Flash | 55 | 55 | length 55 |
| Qwen 3.8 27B | 2 | 2 | length 2 |
| Gemma 4 31B | 0 | 0 | – |
