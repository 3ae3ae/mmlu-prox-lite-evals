"""Run lm-evaluation-harness on MMLU-ProX-Lite for one model config.

    uv run scripts/run_eval.py configs/models/qwen-3-8-27b.yaml --smoke    # 14 requests: English, 1 per subject
    uv run scripts/run_eval.py configs/models/qwen-3-8-27b.yaml            # main run (resumable)
    uv run scripts/run_eval.py configs/models/qwen-3-8-27b.yaml --dry-run  # print the command only

All settings come from configs/protocol.yaml and the model config; this script only turns
them into an `lm-eval run` command.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_config", type=Path)
    parser.add_argument("--smoke", action="store_true", help="English only, 1 question per subject")
    parser.add_argument("--dry-run", action="store_true", help="print the lm-eval command and exit")
    args = parser.parse_args()

    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
    model = yaml.safe_load(args.model_config.read_text())
    provider = protocol["providers"][model["provider"]]

    dataset = protocol["dataset"]
    current = HfApi().dataset_info(dataset["repo"]).sha
    if current != dataset["revision"]:
        sys.exit(f"{dataset['repo']} is at {current}, but the protocol pins {dataset['revision']}. Review before running.")

    languages = ["en"] if args.smoke else protocol["languages"]
    out_dir = ROOT / "runs" / args.model_config.stem / ("smoke" if args.smoke else "main")

    model_args = {
        "model": model["model_id"],
        "base_url": provider["base_url"],
        "num_concurrent": provider["num_concurrent"],
        "max_retries": protocol["run"]["max_retries"],
        "timeout": provider["timeout_s"],
        "tokenized_requests": False,
    }
    gen_kwargs = {
        **model["sampling"],
        "stop": None,
        "max_gen_toks": model["max_tokens"],
        "reasoning_effort": model["reasoning"]["effort"],
    }
    cmd = [
        "lm-eval", "run",
        "--model", "local-chat-completions",
        "--model_args", json.dumps(model_args),
        "--tasks", *[protocol["harness"]["task"].format(lang=lang) for lang in languages],
        "--apply_chat_template",
        "--fewshot_as_multiturn", "false",
        "--gen_kwargs", json.dumps(gen_kwargs),
        "--log_samples",
        "--output_path", str(out_dir),
        "--use_cache", str(out_dir / "cache"),
    ]
    if args.smoke:
        cmd += ["--limit", "1"]

    print(shlex.join(cmd))
    if args.dry_run:
        return 0

    # Non-empty values in .env take precedence over the shell environment.
    env = dict(os.environ)
    env.update({k: v for k, v in dotenv_values(ROOT / ".env").items() if v})
    key = env.get(provider["api_key_env"])
    if not key:
        sys.exit(f"{provider['api_key_env']} is not set (see .env.example).")
    out_dir.mkdir(parents=True, exist_ok=True)
    # local-chat-completions reads its bearer token from OPENAI_API_KEY.
    status = subprocess.call(cmd, env={**env, "OPENAI_API_KEY": key})

    if status == 0 and not args.smoke:
        results_dir = ROOT / protocol["outputs"]["results_dir"].format(model=args.model_config.stem)
        results_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.rglob("results_*.json"):
            shutil.copy2(f, results_dir / f.name)
    return status


if __name__ == "__main__":
    sys.exit(main())
