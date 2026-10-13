"""Exact prompt and answer token counts of every main-run question for the five Neuralwatt models.

    uv run --with transformers --with jinja2 python analysis/usage_recovery/prompt_tokens.py

Uses each served checkpoint's own tokenizer and chat template (data files only; no remote code),
downloaded to runs/usage-recovery/tokenizers/. The counts equal what Neuralwatt reports for the
same request (checked on re-run requests; Gemma's API count is one lower than its template's).
Writes results/recovered-usage/records/prompt_tokens.csv.gz.
"""

import csv
import gzip
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze import finished_models  # noqa: E402

CHECKPOINTS = {"deepseek-v4-1-flash": "deepseek-ai/DeepSeek-V4.1-Flash", "glm-5-3": "zai-org/GLM-5.3",
               "glm-5-3-flash": "zai-org/GLM-5.3-Flash", "qwen-3-8-27b": "Qwen/Qwen3.8-27B-FP8",
               "gemma-4-31b": "nvidia/Gemma-4-31B-IT-NVFP4"}
OFFSET = {"gemma-4-31b": -1}


def main():
    cache = ROOT / "runs/usage-recovery/tokenizers"
    runs = dict(finished_models())
    with gzip.open(ROOT / "results/recovered-usage/records/prompt_tokens.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "language", "subject", "doc_id", "question_id_src", "prompt_tokens", "answer_tokens"])
        for stem, repo in CHECKPOINTS.items():
            local = cache / stem
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                if not (local / name).exists():
                    hf_hub_download(repo, name, local_dir=local)
            tokenizer = AutoTokenizer.from_pretrained(local)
            path = runs[stem]
            stamp = path.stem.removeprefix("results_")
            for sample_file in sorted(path.parent.glob(f"samples_mmlu_prox_lite_*_{stamp}.jsonl")):
                language = sample_file.name.removeprefix("samples_mmlu_prox_lite_")[:2]
                for line in sample_file.open(encoding="utf-8"):
                    sample = json.loads(line)
                    prompt = sample["arguments"]["gen_args_0"]["arg_0"]
                    messages = json.loads(prompt[0] if isinstance(prompt, list) else prompt)
                    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
                    n = len(ids["input_ids"] if hasattr(ids, "keys") else ids) + OFFSET.get(stem, 0)
                    answer = len(tokenizer(sample["resps"][0][0] or "", add_special_tokens=False)["input_ids"])
                    writer.writerow([stem, language, sample["doc"]["category"], sample["doc_id"],
                                     sample["doc"]["question_id_src"], n, answer])
            print(stem, "done")


if __name__ == "__main__":
    main()
