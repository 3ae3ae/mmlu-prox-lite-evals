"""Re-score strict-wrong answers from finished main runs with the Decisions API."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import sys
from threading import Event
from time import perf_counter

import requests
import yaml

from analyze import ROOT, finished_models, load_samples


GRADER = "gpt-6-luna"
INSTRUCTIONS = (
    "This response answers a multiple-choice question. Which option does the response itself "
    "give as its final answer? Judge only what the response says, not which option is correct. "
    'Choose "none" if the response gives no final answer or is cut off before giving one.'
)


def request_body(row):
    choices = [
        {"value": chr(ord("A") + index), "description": row["doc"][f"option_{index}"]}
        for index in range(10)
        if row["doc"][f"option_{index}"] not in ("N/A", "", None)
    ]
    choices.append({"value": "none", "description": "The response gives no final answer."})
    return {"model": GRADER, "input": "Response:\n" + (row["response_text"] or ""),
            "questions": [{"type": "choice", "name": "final_answer",
                           "instructions": INSTRUCTIONS, "choices": choices}]}


def grade_row(row, api_key, stopped):
    body = request_body(row)
    try:
        for attempt in range(6):
            if stopped.is_set():
                return None
            try:
                response = requests.post(
                    "https://api.openai.com/v1/decisions",
                    headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=60,
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 5:
                    raise RuntimeError("Decisions API connection/timeout error after 6 attempts") from None
            else:
                if response.ok:
                    break
                if response.status_code != 429 and not 500 <= response.status_code < 600:
                    raise RuntimeError(f"Decisions API HTTP {response.status_code}: {response.text}")
                if attempt == 5:
                    raise RuntimeError(f"Decisions API HTTP {response.status_code} after 6 attempts: {response.text}")
            if stopped.wait(2 ** attempt):
                return None

        answers = response.json()["answers"]
        if len(answers) != 1 or answers[0].get("name") != "final_answer":
            raise ValueError("Decisions API returned an unexpected answers array")
        answer = answers[0]
        refusal = answer["type"] == "refusal"
        if not refusal:
            allowed = {choice["value"] for choice in body["questions"][0]["choices"]}
            if answer["type"] != "choice" or answer["choice"] not in allowed:
                raise ValueError("Decisions API returned an unexpected choice")
            probabilities = {item["value"]: item["probability"] for item in answer["probabilities"]}
        return {
            "language": row["language"], "subject": row["subject"],
            "question_id_src": row["question_id_src"], "gold": row["gold"],
            "strict_pred": row["strict_pred"], "choice": None if refusal else answer["choice"],
            "confidence": None if refusal else answer["confidence"],
            "p_gold": None if refusal else probabilities[row["gold"]],
            "refusal": refusal, "grader": GRADER,
            "graded_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        stopped.set()
        raise


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--models", help="Comma-separated model stems")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 0):
        parser.error("--workers must be positive and --limit must be nonnegative")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY is not set in the environment")

    languages = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text(encoding="utf-8"))["languages"]
    models = set(args.models.split(",")) if args.models else None
    output_dir = ROOT / "results/lenient"
    jobs = []
    skipped = graded = 0
    for stem, result_path in finished_models():
        if models is not None and stem not in models:
            continue
        path = output_dir / f"{stem}.jsonl"
        seen = set()
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    result = json.loads(line)
                    seen.add((result["language"], result["question_id_src"]))
        data, _ = load_samples(result_path, stem, languages, include_responses=True)
        for row in data.loc[data["exact_match"] == 0].to_dict("records"):
            key = (row["language"], row["question_id_src"])
            if key in seen:
                skipped += 1
                continue
            if args.limit is None or len(jobs) < args.limit:
                jobs.append((stem, row))
                seen.add(key)

    stopped = Event()
    streams = {}
    error = None
    executor = ThreadPoolExecutor(max_workers=args.workers)
    try:
        if jobs:
            output_dir.mkdir(parents=True, exist_ok=True)
            futures = {executor.submit(grade_row, row, api_key, stopped): stem for stem, row in jobs}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as failure:
                    stopped.set()
                    if error is None:
                        error = failure
                        print(str(failure).replace(api_key, "[redacted]"), file=sys.stderr)
                    continue
                if result is None:
                    continue
                stem = futures[future]
                if stem not in streams:
                    streams[stem] = (output_dir / f"{stem}.jsonl").open("a", encoding="utf-8")
                streams[stem].write(json.dumps(result, ensure_ascii=False) + "\n")
                streams[stem].flush()
                graded += 1
    finally:
        # On Ctrl-C or an error, drop queued requests instead of sending them with nobody saving results.
        stopped.set()
        executor.shutdown(wait=True, cancel_futures=True)
        for stream in streams.values():
            stream.close()
        print(f"Graded {graded} rows; skipped {skipped} already graded; elapsed {perf_counter() - started:.2f} seconds.")
    return 1 if error is not None else 0


if __name__ == "__main__":
    sys.exit(main())
