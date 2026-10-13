"""Tie the providers' usage records of the main run to its questions.

    uv run python analysis/usage_recovery/match.py

The usage records hold no prompt or answer text, so each is matched by two facts:
- lm-eval sends a model's requests in a fixed order (language, then subject A-Z, then question),
  so records sorted by start time follow it; resumed runs restart from the first unfinished one.
- a record's prompt token count must equal the question's. It is known exactly for the Neuralwatt
  models (their tokenizers, prompt_tokens.py) and the GPT models (o200k + 10). For Claude and Gemini
  it is known only for the questions of a usage-logged re-run (probe-anchors.csv), which anchor an
  order alignment.
The last record of a question is the answer lm-eval kept. Checks are printed per model.

Inputs (results/recovered-usage/records/): neuralwatt-rows.jsonl.gz, neuralwatt-probe-rows.jsonl.gz,
llmgateway-logs.json.gz (fetch_neuralwatt.py, llmgateway_logs.js), prompt_tokens.csv.gz, probe-anchors.csv.
Outputs: results/recovered-usage/{neuralwatt,llmgateway}.csv and checks.json.
"""

import csv
import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import tiktoken
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze import finished_models  # noqa: E402

IN = ROOT / "results/recovered-usage/records"
OUT = ROOT / "results/recovered-usage"
LANGUAGES = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())["languages"]
NEURALWATT = {"deepseek-v4-1-flash": {"deepseek-v4.1-flash-flex"}, "glm-5-3": {"glm-5.3", "glm-5.3-flex"},
              "glm-5-3-flash": {"glm-5.3-flash-flex"}, "qwen-3-8-27b": {"qwen-3.8-27b", "qwen-3.8-27b-flex"},
              "gemma-4-31b": {"gemma-4-31b"}}
GATEWAY = {"claude-haiku-5-5": "claude-haiku-5-5", "gpt-6-luna": "gpt-6-luna", "gemini-3-8-flash": "gemini-3.8-flash",
           "claude-sonnet-5-5": "claude-sonnet-5-5", "gpt-6-1-sol": "gpt-6.1-sol"}
EXACT_O200K = {"gpt-6-luna": 10, "gpt-6-1-sol": 10}  # prompt tokens = o200k count of the message texts + 10
OVERHEAD = {"gemma-4-31b": 5}  # completion - answer - reasoning tokens on the re-run's requests; 2 for the others
CHECKS = {}
PROBE_START = datetime.fromisoformat("2026-10-13T09:10:00+00:00").timestamp()
O200K = tiktoken.get_encoding("o200k_base")


def read(name):
    return gzip.open(IN / name, "rt", encoding="utf-8") if name.endswith(".gz") else (IN / name).open(encoding="utf-8")


def probe_anchors(stem):
    """(language, question_id_src) -> (prompt tokens, request id) of the re-run's requests."""
    return {(r["language"], r["question_id_src"]): (int(r["prompt_tokens"]), r["request_id"])
            for r in csv.DictReader(read("probe-anchors.csv")) if r["model"] == stem}


def questions(stem):
    """The main run's questions in dispatch order, with what lm-eval logged."""
    doc_ids = {(r["language"], r["question_id_src"]): int(r["doc_id"])
               for r in csv.DictReader(read("prompt_tokens.csv.gz"))}
    path = dict(finished_models())[stem]
    stamp = path.stem.removeprefix("results_")
    items = []
    for sample_file in path.parent.glob(f"samples_mmlu_prox_lite_*_{stamp}.jsonl"):
        language = sample_file.name.removeprefix("samples_mmlu_prox_lite_")[:2]
        subject = sample_file.name.removeprefix(f"samples_mmlu_prox_lite_{language}_").removesuffix(f"_{stamp}.jsonl")
        for line in sample_file.open(encoding="utf-8"):
            sample = json.loads(line)
            prompt = sample["arguments"]["gen_args_0"]["arg_0"]
            messages = json.loads(prompt[0] if isinstance(prompt, list) else prompt)
            qid = str(sample["doc"]["question_id_src"])
            items.append({"language": language, "subject": subject, "question_id_src": qid,
                          "doc_id": doc_ids[(language, qid)], "correct": int(float(sample["exact_match"])),
                          "empty_answer": int(not (sample["resps"][0][0] or "").strip()),
                          "o200k": sum(len(O200K.encode(m["content"], disallowed_special=())) for m in messages)})
    assert len(items) == 5880, f"{stem}: expected 5880 questions"
    return sorted(items, key=lambda q: (LANGUAGES.index(q["language"]), q["subject"], q["doc_id"]))


def greedy(tokens, records, forward=60, back=30):
    """Match records (sorted by start) to question indices with equal prompt tokens near a moving pointer."""
    by_tokens = defaultdict(list)
    for j, t in enumerate(tokens):
        by_tokens[t].append(j)
    seen, pointer, pairs = [False] * len(tokens), -1, []
    for record in sorted(records, key=lambda r: r["start"]):
        candidates = by_tokens.get(record["prompt_tokens"], [])
        near = [j for j in candidates if pointer - back <= j <= pointer + forward]
        if near:  # prefer a question with no record yet, then the next expected position
            j = min(near, key=lambda j: (seen[j], abs(j - pointer - 1)))
        elif candidates:  # a resumed run starts again from the earliest question still without a record
            fresh = [j for j in candidates if not seen[j]]
            j = min(fresh) if fresh else max(candidates)
        else:
            pairs.append((record, None))
            continue
        seen[j], pointer = True, j
        pairs.append((record, j))
    return pairs


def anchored(tokens, predicted, tolerance, records, band=500, skip_record=-2.0, skip_question=-4.0):
    """Monotone alignment (DP) of records to questions; exact tokens where known, else a tolerance."""
    records = sorted(records, key=lambda r: r["start"])
    n, m, width = len(tokens), len(records), 2 * band + 1
    observed = [r["prompt_tokens"] for r in records]
    score = np.full((n + 1, width), -1e18)
    move = np.zeros((n + 1, width), dtype=np.int8)  # 0 match, 1 skip record, 2 skip question
    center = [round(i * m / n) for i in range(n + 1)]
    col = lambda i, k: k - center[i] + band
    score[0, col(0, 0)] = 0
    for k in range(1, min(m, band) + 1):
        score[0, col(0, k)], move[0, col(0, k)] = k * skip_record, 1
    for i in range(1, n + 1):
        exact, guess, tol = tokens[i - 1], predicted[i - 1], tolerance[i - 1]
        for k in range(max(0, center[i] - band), min(m, center[i] + band) + 1):
            best, how = -1e18, 0
            if k > 0:
                c = col(i - 1, k - 1)
                if 0 <= c < width and score[i - 1, c] > -1e18:
                    o = observed[k - 1]
                    gain = (20.0 if o == exact else None) if exact is not None else \
                        (5.0 - 2.0 * abs(o - guess) / tol if abs(o - guess) <= tol else None)
                    if gain is not None and score[i - 1, c] + gain > best:
                        best, how = score[i - 1, c] + gain, 0
                c = col(i, k - 1)
                if 0 <= c < width and score[i, c] + skip_record > best:
                    best, how = score[i, c] + skip_record, 1
            c = col(i - 1, k)
            if 0 <= c < width and score[i - 1, c] + skip_question > best:
                best, how = score[i - 1, c] + skip_question, 2
            score[i, col(i, k)], move[i, col(i, k)] = best, how
    k = max(range(max(0, center[n] - band), min(m, center[n] + band) + 1),
            key=lambda k: score[n, col(n, k)] + skip_record * (m - k))  # trailing records: re-sends
    pairs, i = {}, n
    while i > 0:
        how = move[i, col(i, k)] if k > 0 else 2
        if how == 0:
            pairs[i - 1] = k - 1
            i, k = i - 1, k - 1
        elif how == 1:
            k -= 1
        else:
            i -= 1
    return records, pairs


def ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def neuralwatt():
    exact = defaultdict(dict)
    for r in csv.DictReader(read("prompt_tokens.csv.gz")):
        exact[r["model"]][(r["language"], r["question_id_src"])] = (int(r["prompt_tokens"]), int(r["answer_tokens"]))
    rows = [json.loads(line) for line in read("neuralwatt-rows.jsonl.gz")]
    probe_rows = [json.loads(line) for line in read("neuralwatt-probe-rows.jsonl.gz")]
    for r in rows + probe_rows:
        r["start"] = ts(r["created_at"]) - (r["duration"] or 0)
    out = []
    for stem, names in NEURALWATT.items():
        items = questions(stem)
        tokens = [exact[stem][(q["language"], q["question_id_src"])][0] for q in items]
        answer = [exact[stem][(q["language"], q["question_id_src"])][1] for q in items]
        mine = [r for r in rows if r.get("requested_model") in names and ts(r["created_at"]) < PROBE_START]
        per = defaultdict(list)
        pairs = greedy(tokens, mine)
        for record, j in pairs:
            if j is not None:
                per[j].append(record)
        # check: the re-run's requests, whose question is known from their request ID
        truth = {rid: question for question, (_, rid) in probe_anchors(stem).items()}
        asked = set(truth.values())
        probe_items = [j for j, q in enumerate(items) if (q["language"], q["question_id_src"]) in asked]
        probe = [r for r in probe_rows if r["request_id"] in truth]
        hits = sum(1 for r, j in greedy([tokens[i] for i in probe_items], probe)
                   if j is not None and (items[probe_items[j]]["language"], items[probe_items[j]]["question_id_src"])
                   == truth[r["request_id"]])
        CHECKS[stem] = {"matched": len(per), "method": "exact prompt tokens", "check": "re-run requests",
                        "right": hits, "checked": len(probe)}
        print(f"{stem}: {len(per)}/5880 questions matched, {sum(j is None for _, j in pairs)} records unmatched; "
              f"re-run check {hits}/{len(probe)} right")
        for j, q in enumerate(items):
            row = {k: q[k] for k in ("language", "subject", "question_id_src", "correct", "empty_answer")}
            row.update(model=stem, prompt_tokens=tokens[j], answer_tokens=answer[j])
            if per[j]:
                r = max(per[j], key=lambda r: r["created_at"])
                row.update(cached_tokens=r["cached_tokens"], completion_tokens=r["completion_tokens"],
                           reasoning_tokens_est=max(r["completion_tokens"] - answer[j] - OVERHEAD.get(stem, 2), 0),
                           finish_reason=r["finish_reason"], duration_s=round(r["duration"], 1),
                           energy_joules=r["energy_joules"], cost_usd=r["cost_usd"],
                           requested_model=r["requested_model"], attempts=len(per[j]), request_id=r["request_id"])
            out.append(row)
    write(OUT / "neuralwatt.csv", out, ["model", "language", "subject", "question_id_src", "correct", "empty_answer",
                                       "prompt_tokens", "cached_tokens", "completion_tokens", "answer_tokens",
                                       "reasoning_tokens_est", "finish_reason", "duration_s", "energy_joules",
                                       "cost_usd", "requested_model", "attempts", "request_id"])


def anchors_for(stem, items):
    """The re-run questions' exact prompt tokens, by question index."""
    anchors = probe_anchors(stem)
    return {j: anchors[(q["language"], q["question_id_src"])][0] for j, q in enumerate(items)
            if (q["language"], q["question_id_src"]) in anchors}


def predict(items, anchors):
    """Exact tokens at anchors; elsewhere a per-language linear fit on o200k counts, with its worst error."""
    x = np.array([q["o200k"] for q in items], float)
    languages = np.array([q["language"] for q in items])
    predicted, tolerance = np.zeros(len(items)), np.zeros(len(items))
    for language in LANGUAGES:
        idx = np.where(languages == language)[0]
        fit = [j for j in idx if j in anchors]
        if len(fit) < 10:  # language the re-run did not reach: fit on all anchors
            fit = list(anchors)
        a = np.c_[x[fit], np.ones(len(fit))]
        y = np.array([anchors[j] for j in fit], float)
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        predicted[idx] = x[idx] * coef[0] + coef[1]
        tolerance[idx] = np.abs(y - a @ coef).max() * 1.25 + 5
    return [anchors.get(j) for j in range(len(items))], predicted, tolerance


def llmgateway():
    logs = json.loads(read("llmgateway-logs.json.gz").read())
    out = []
    for stem, requested in GATEWAY.items():
        items = questions(stem)
        effort = yaml.safe_load((ROOT / "configs/models" / f"{stem}.yaml").read_text())["reasoning"]["effort"]
        # completed attempts of the main run; other uses of the account differ in reasoning effort
        records = [dict(r, prompt_tokens=int(r["promptTokens"] or -1), start=ts(r["createdAt"]) - int(r["duration"] or 0) / 1000)
                   for r in logs if r["requestedModel"] == requested and r["hasError"] in (False, "false")
                   and r["finishReason"] != "canceled" and r["reasoningEffort"] == effort]
        per = defaultdict(list)
        if stem in EXACT_O200K:
            method = "exact prompt tokens"
            for record, j in greedy([q["o200k"] + EXACT_O200K[stem] for q in items], records):
                if j is not None:
                    per[j].append(record)
            check = ""
            CHECKS[stem] = {"method": method}
        else:
            method = "anchored order"
            anchors = anchors_for(stem, items)
            rng = np.random.default_rng(0)
            held = {j for j in anchors if rng.random() < 0.3}
            ordered, pairs = anchored(*predict(items, {j: t for j, t in anchors.items() if j not in held}), records)
            hits = sum(1 for j in held if j in pairs and ordered[pairs[j]]["prompt_tokens"] == anchors[j])
            check = f"; held-out anchors {hits}/{len(held)} right"
            CHECKS[stem] = {"method": method, "check": "held-out anchors", "right": hits, "checked": len(held)}
            ordered, pairs = anchored(*predict(items, anchors), records)
            used = set(pairs.values())
            by_tokens = defaultdict(list)
            for k, r in enumerate(ordered):
                if k not in used:
                    by_tokens[r["prompt_tokens"]].append(r)
            for j, k in pairs.items():  # unmatched attempts of the same prompt nearby or later: retries, re-sends
                per[j] = [ordered[k]] + [r for r in by_tokens.get(ordered[k]["prompt_tokens"], [])
                                         if r["start"] > ordered[max(k - 15, 0)]["start"]]
        CHECKS[stem]["matched"] = sum(1 for v in per.values() if v)
        print(f"{stem}: {CHECKS[stem]['matched']}/5880 questions matched ({method}){check}")
        for j, q in enumerate(items):
            row = {k: q[k] for k in ("language", "subject", "question_id_src", "correct", "empty_answer")}
            row.update(model=stem, match_method=method)
            if per[j]:
                r = max(per[j], key=lambda r: r["start"])
                row.update(prompt_tokens=r["prompt_tokens"], completion_tokens=r["completionTokens"],
                           reasoning_tokens=r["reasoningTokens"], finish_reason=r["finishReason"],
                           duration_s=int(r["duration"] or 0) / 1000, cost_usd=r["cost"], served_by=r["usedProvider"],
                           attempts=len(per[j]), log_id=r["id"])
            out.append(row)
    write(OUT / "llmgateway.csv", out, ["model", "language", "subject", "question_id_src", "correct", "empty_answer",
                                       "prompt_tokens", "completion_tokens", "reasoning_tokens", "finish_reason",
                                       "duration_s", "cost_usd", "served_by", "attempts", "log_id", "match_method"])


def write(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    neuralwatt()
    llmgateway()
    (OUT / "checks.json").write_text(json.dumps(CHECKS, indent=1) + "\n")
