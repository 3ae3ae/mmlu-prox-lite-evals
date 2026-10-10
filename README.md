# MMLU-ProX-Lite: Where Do Recent Lightweight LLMs Break Down?

Per-language and per-subject evaluation of recent LLMs on
[MMLU-ProX-Lite](https://huggingface.co/datasets/li-lab/MMLU-ProX-Lite) (588 parallel questions × 14 subjects),
in 10 of its 29 languages, run with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

> **Status:** protocol v1 fixed; runs not started yet.

## Purpose

This is **not a leaderboard**. The core comparisons are across languages and across subjects
*within* each model:

- **Language gap:** how much each model's accuracy drops in each language relative to its own
  English accuracy, measured on the same (parallel) questions.
- **Subject profile:** which subjects each model is weak in, overall and per language.
- **Exploratory:** language × subject combinations, and whether a low score comes from wrong
  answers or from not following the answer format in that language.

Models are deliberately **not** ranked against each other: each runs at its own developer's
default reasoning effort, so differences between models mix model quality with reasoning budget.
The analyses were fixed in [`configs/protocol.yaml`](configs/protocol.yaml) before any results were seen.

## Results

_TBD._ Raw per-question outputs: _TBD (Hugging Face dataset link)_.

## Models

Setup verified on 2026-10-09. Full settings with sources are in [`configs/models/`](configs/models).

| Model | Served via | Served checkpoint | Reasoning effort | Max tokens |
|---|---|---|---|---|
| Claude Haiku 5.5 | LLM Gateway (DevPass) | `claude-haiku-5-5` | `medium` | 64,000 |
| GPT-6 Luna | LLM Gateway (DevPass) | `gpt-6-luna` | `medium` | 64,000 |
| Gemini 3.8 Flash | LLM Gateway (DevPass) | `gemini-3.8-flash` | `medium` | 65,536 |
| Claude Sonnet 5.5 † | LLM Gateway (DevPass) | `claude-sonnet-5-5` | `high` | 64,000 |
| GPT-6.1 Sol † | LLM Gateway (DevPass) | `gpt-6.1-sol` | `medium` | 64,000 |
| DeepSeek V4.1 Flash | Neuralwatt (flex) | `deepseek-ai/DeepSeek-V4.1-Flash` | `high` | 64,000 |
| GLM-5.3 † | Neuralwatt (flex) | `zai-org/GLM-5.3` | `max` | 64,000 |
| GLM-5.3-Flash | Neuralwatt (flex) | `zai-org/GLM-5.3-Flash` | `max` | 64,000 |
| Qwen 3.8 27B | Neuralwatt (flex) | `Qwen/Qwen3.8-27B-FP8` (FP8) | `xhigh` | 64,000 |
| Gemma 4 31B | Neuralwatt | `nvidia/Gemma-4-31B-IT-NVFP4` (4-bit) | `max` (thinking on) | 16,384 |

† Larger mid-tier or flagship models included as references; the others are lightweight models.

Closed models go through [LLM Gateway](https://llmgateway.io), which routes each request to one of
the model developer's own or cloud-partner endpoints (e.g. Anthropic or AWS Bedrock for Claude).
Open-weight models are served by [Neuralwatt](https://portal.neuralwatt.com/), which publishes the
exact Hugging Face checkpoint (including quantization) it serves. Quantization can hurt
non-Latin-script languages more than English ([Marchisio et al., 2024](https://arxiv.org/abs/2407.03211)),
so results for quantized checkpoints describe that checkpoint, not the full-precision model.

## Methodology (protocol v1)

The full protocol is in [`configs/protocol.yaml`](configs/protocol.yaml) and was fixed before any main-run results were seen.

- **Harness:** lm-evaluation-harness `0.4.13`, tasks `mmlu_prox_lite_{lang}` unchanged, backend
  `local-chat-completions` (OpenAI-compatible chat API) for every model,
  `--apply_chat_template --fewshot_as_multiturn false`.
- **Data:** `li-lab/MMLU-ProX-Lite` at revision `e82aafb`, test split (588 questions per language).
- **Languages:** `en`, `zh`, `ja`, `ko`, `pt`, `de`, `id`, `ru`, `ar`, `sw`.
- **Prompt:** the task's 5-shot chain-of-thought prompt: a system message with the task
  description and one user message with 5 worked examples from the same subject plus the test question.
- **Changes to the task's generation defaults** (same for every model):
  - temperature `0.0` → `1.0`: every model's developer default or recommendation, and several APIs
    reject or ignore other values;
  - stop strings → none: chat models end their turn by themselves, and a stop string such as
    `Q:` inside the reasoning would cut it off;
  - `max_gen_toks` `2048` → the per-model value above, since reasoning tokens count toward it.
- **Reasoning effort:** each developer's official API default; otherwise the model card's recommendation;
  otherwise the host's default. Exception: small open models (~30B: Gemma 4 31B, Qwen 3.8 27B) use their highest effort.
- **Sampling:** each developer's recommended values (see the model configs).
- **Scoring:** the official score is lm-eval's `exact_match` with the task's per-language answer regex
  (`custom-extract`); unextractable answers count as wrong. A lenient re-scoring (accepting any
  language's answer phrase or a final standalone `(A)`–`(J)`) can be computed from the logged raw
  responses to separate format-following failures from wrong answers.
- **Statistics:** language gaps are paired comparisons on the same questions (paired bootstrap CIs).
  Language × subject cells use Benjamini–Hochberg correction.

## Limitations

- Lite subjects are unbalanced (per language: 19 history … 68 math questions). Subject-level
  95% CIs can be as wide as ±22 points; treat single-cell differences as exploratory.
- lm-eval logs only the response text, so truncated responses cannot be told apart from wrong
  answers in the published logs. Max tokens are set generously to keep truncation rare. For
  Neuralwatt-served models, the per-run truncation rate is available from Neuralwatt's usage API
  (`finish_reason` per request), but it cannot be linked to individual questions.
- Averages are over the 10 evaluated languages and are not comparable to 29-language averages reported elsewhere.
- API models change behind stable names; result files carry their run timestamp.
- Reasoning settings differ across models by design, so cross-model comparisons are not meaningful.

## Repository layout

```
configs/protocol.yaml    evaluation protocol v1
configs/models/          one file per model: ID, reasoning effort, sampling, max tokens, sources
scripts/run_eval.py      turns a model config into an `lm-eval run` command and runs it
analysis/                analysis scripts and figures
results/                 aggregated tables (committed)
runs/                    raw lm-eval outputs (git-ignored, published to Hugging Face)
hf/                      Hugging Face dataset card for the raw outputs
```

## Reproduce

```bash
uv sync
cp .env.example .env   # LLM_GATEWAY_API_KEY, NEURALWATT_API_KEY, HF_TOKEN (write access)

# 14 requests (English, 1 question per subject) to check the API accepts every parameter
uv run scripts/run_eval.py configs/models/qwen-3-8-27b.yaml --smoke

# main run; rerunning the same command resumes from the request cache.
# Uploads results and samples to a private Hugging Face dataset and copies
# results_*.json into results/<model>/.
uv run scripts/run_eval.py configs/models/qwen-3-8-27b.yaml
```

`--dry-run` prints the exact `lm-eval run` command without sending requests.

## Citation

This project evaluates models on MMLU-ProX. Please cite the original benchmark:

```bibtex
@article{xuan2025mmlu,
  title={Mmlu-prox: A multilingual benchmark for advanced large language model evaluation},
  author={Weihao Xuan and Rui Yang and Heli Qi and Qingcheng Zeng and Yunze Xiao and Aosong Feng and Dairui Liu and Yun Xing and Junjue Wang and Fan Gao and Jinghui Lu and Yuang Jiang and Huitao Li and Xin Li and Kunyu Yu and Ruihai Dong and Shangding Gu and Yuekang Li and Xiaofei Xie and Felix Juefei-Xu and Foutse Khomh and Osamu Yoshie and Qingyu Chen and Douglas Teodoro and Nan Liu and Randy Goebel and Lei Ma and Edison Marrese-Taylor and Shijian Lu and Yusuke Iwasawa and Yutaka Matsuo and Irene Li},
  journal={arXiv preprint arXiv:2503.10497},
  year={2025}
}
```

## License

Code and results: [MIT](LICENSE).

Third-party material:

- **MMLU-ProX** (questions included in the logged samples): MIT License, as stated on its
  [dataset card](https://huggingface.co/datasets/li-lab/MMLU-ProX-Lite). It is built on
  [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (MIT).
- **lm-evaluation-harness**: MIT License; used as an unmodified dependency, not redistributed here.
- **Model outputs** in the logs were generated by the models listed above and are shared for
  evaluation analysis.
