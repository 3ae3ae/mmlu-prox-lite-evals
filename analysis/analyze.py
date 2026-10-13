"""Summarize finished main runs and update a marked README results block."""

import argparse
from datetime import date
import json
from math import comb
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd
import yaml


ORDER = [
    "claude-haiku-5-5",
    "gpt-6-luna",
    "gemini-3-8-flash",
    "claude-sonnet-5-5",
    "gpt-6-1-sol",
    "deepseek-v4-1-flash",
    "glm-5-3",
    "glm-5-3-flash",
    "qwen-3-8-27b",
    "gemma-4-31b",
]
REFERENCE = {"claude-sonnet-5-5", "gpt-6-1-sol", "glm-5-3"}
ROOT = Path(__file__).resolve().parents[1]


def finished_models():
    for stem in ORDER:
        result_paths = list((ROOT / "runs" / stem / "main").glob("*/results_*.json"))
        if result_paths:
            yield stem, max(result_paths, key=lambda path: path.name)


def load_samples(result_path, stem, languages, include_responses=False):
    timestamp = result_path.stem.removeprefix("results_")
    paths = sorted(result_path.parent.glob(f"samples_mmlu_prox_lite_*_{timestamp}.jsonl"))
    assert len(paths) == 140, f"{stem}: expected 140 sample files, found {len(paths)}"
    pattern = re.compile(
        rf"samples_mmlu_prox_lite_([a-z]{{2}})_(.+)_{re.escape(timestamp)}\.jsonl"
    )
    records = []
    pairs = set()
    for path in paths:
        match = pattern.fullmatch(path.name)
        assert match, f"{stem}: malformed sample filename: {path.name}"
        language, subject = match.groups()
        assert language in languages, f"{stem}: unexpected language {language}"
        pairs.add((language, subject))
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                score = row["exact_match"]
                assert score in (0.0, 1.0), (
                    f"{path}:{line_number}: exact_match must be 0 or 1, found {score!r}"
                )
                records.append({
                    "language": language,
                    "subject": subject,
                    "question_id_src": row["doc"]["question_id_src"],
                    "exact_match": score,
                    "invalid": row["filtered_resps"][0] == "[invalid]",
                    "gold": row["doc"]["answer"],
                    "strict_pred": row["filtered_resps"][0],
                })
                if include_responses:
                    records[-1].update(doc=row["doc"], response_text=row["resps"][0][0])
    subjects = {subject for _, subject in pairs}
    assert len(subjects) == 14 and pairs == {
        (language, subject) for language in languages for subject in subjects
    }, f"{stem}: expected exactly 10 languages × 14 matching subjects"
    data = pd.DataFrame(records)
    assert len(data) == 5880, f"{stem}: expected 5880 rows, found {len(data)}"
    baseline_ids = set(data.loc[data["language"] == "en", "question_id_src"])
    results = json.loads(result_path.read_text(encoding="utf-8"))["results"]
    for language in languages:
        rows = data.loc[data["language"] == language]
        assert len(rows) == 588, f"{stem}/{language}: expected 588 rows, found {len(rows)}"
        assert not rows["question_id_src"].duplicated().any(), (
            f"{stem}/{language}: duplicate question_id_src values"
        )
        assert set(rows["question_id_src"]) == baseline_ids, (
            f"{stem}/{language}: question_id_src set differs from English"
        )
        actual = rows["exact_match"].mean()
        expected = results[f"mmlu_prox_lite_{language}"]["exact_match,custom-extract"]
        assert abs(actual - expected) <= 1e-9, (
            f"{stem}/{language}: sample accuracy {actual:.12f} differs from "
            f"results JSON accuracy {expected:.12f} (tolerance 1e-9)"
        )
    counts = data.groupby(["subject", "language"])["question_id_src"].nunique().unstack()
    assert counts.eq(counts["en"], axis=0).all().all(), (
        f"{stem}: subject question counts differ across languages"
    )
    return data, counts["en"].to_dict()


def load_lenient(stem, data):
    path = ROOT / "results/lenient" / f"{stem}.jsonl"
    expected = set(data.loc[data["exact_match"] == 0, ["language", "question_id_src"]]
                   .itertuples(index=False, name=None))
    records = []
    try:
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream]
        keys = [(row["language"], row["question_id_src"]) for row in records]
        if not path.exists() or set(keys) != expected or len(keys) != len(set(keys)):
            print(f"{stem}: incomplete lenient coverage ({len(keys)} graded, "
                  f"{len(expected)} strict-wrong; keys must match without duplicates); "
                  "omitted from lenient tables.")
            return None
        grades = pd.DataFrame(records, columns=["language", "question_id_src", "choice", "refusal"])
        scored = data.merge(grades, on=["language", "question_id_src"], how="left", validate="one_to_one")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"{stem}: incomplete lenient coverage ({error}); omitted from lenient tables.")
        return None
    scored["rescued"] = (scored["exact_match"].eq(0) & scored["refusal"].eq(False)
                         & scored["choice"].eq(scored["gold"]))
    scored["lenient_correct"] = scored["exact_match"].eq(1) | scored["rescued"]
    return scored


def markdown_table(headers, rows):
    def line(cells):
        return "| " + " | ".join(map(str, cells)) + " |"

    return "\n".join([line(headers), line(["---"] * len(headers))] + [line(row) for row in rows])


def mcnemar(b, c):
    discordant = b + c
    if discordant == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(discordant, k) for k in range(min(b, c) + 1)) / 2 ** discordant)


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1)
    q_values = np.empty_like(p_values)
    q_values[order] = adjusted
    return q_values


def readme_sections(path, name="results"):
    # Work in bytes so text and line endings outside the block stay untouched.
    lines = path.read_bytes().splitlines(keepends=True)
    positions = []
    for marker in (f"<!-- {name}:start -->".encode(), f"<!-- {name}:end -->".encode()):
        matches = [i for i, line in enumerate(lines) if line.rstrip(b"\r\n") == marker]
        if len(matches) != 1:
            raise ValueError(f"{path}: expected exactly one marker line {marker.decode()}, found {len(matches)}")
        positions.append(matches[0])
    start, end = positions
    if start >= end:
        raise ValueError(f"{path}: {name}:start must precede {name}:end")
    return b"".join(lines[:start + 1]), b"".join(lines[end:])


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = parser.parse_args()
    try:
        prefix, suffix = readme_sections(args.readme)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text(encoding="utf-8"))
    languages = protocol["languages"]
    assert len(languages) == 10 and languages[0] == "en" and len(set(languages)) == 10, (
        "Protocol must list 10 distinct languages, with English first"
    )
    resamples = protocol["analysis"]["bootstrap_resamples"]
    assert mcnemar(0, 5) == 0.0625
    assert mcnemar(3, 3) == 1.0
    assert mcnemar(0, 0) == 1.0
    assert np.allclose(benjamini_hochberg([0.01, 0.04, 0.03, 0.2]),
                       [0.04, 0.04 / 0.75, 0.04 / 0.75, 0.2], rtol=0, atol=1e-9)
    rng = np.random.default_rng(20261012)
    cell_rng = np.random.default_rng(20261012)
    accuracy_records, gap_records, subject_records, failure_records = [], [], [], []
    accuracy_rows, gap_rows, compact_rows, subject_rows, failure_rows = [], [], [], [], []
    lenient_records, rescue_records = [], []
    lenient_rows, gain_rows, rescue_rows = [], [], []
    cell_accuracy_records, cell_gap_records = [], []
    cell_accuracy_tables, cell_gap_rows, cell_compact_rows = [], [], []
    included = []
    question_counts = None

    for stem, result_path in finished_models():
        data, counts = load_samples(result_path, stem, languages)
        if question_counts is None:
            question_counts = counts
        assert counts == question_counts, f"{stem}: subject question counts differ from other models"
        name = yaml.safe_load((ROOT / "configs/models" / f"{stem}.yaml").read_text(encoding="utf-8"))["name"]
        display = name + (" †" if stem in REFERENCE else "")
        included.append(stem)

        accuracies = data.groupby("language")["exact_match"].mean() * 100
        failures = data.groupby("language")["invalid"].sum()
        accuracy_rows.append([display] + [f"{accuracies[lang]:.1f}" for lang in languages]
                             + [f"{data['exact_match'].mean() * 100:.1f}"])
        failure_rows.append([display] + [int(failures[lang]) for lang in languages])
        for language in languages:
            accuracy_records.append({"model": stem, "language": language, "accuracy": accuracies[language]})
            failure_records.append({"model": stem, "language": language, "invalid": int(failures[language])})

        scored = load_lenient(stem, data)
        if scored is not None:
            lenient = scored.groupby("language")["lenient_correct"].mean() * 100
            assert (lenient >= accuracies).all(), f"{stem}: lenient accuracy fell below strict accuracy"
            gain = lenient - accuracies
            lenient_rows.append([display] + [f"{lenient[lang]:.1f}" for lang in languages]
                                + [f"{scored['lenient_correct'].mean() * 100:.1f}"])
            gain_rows.append([display] + [f"{gain[lang]:+.1f}" for lang in languages])
            for language in languages:
                rows = scored.loc[scored["language"] == language]
                lenient_records.append({"model": stem, "language": language,
                                        "strict": accuracies[language], "lenient": lenient[language],
                                        "gain_pp": gain[language]})
                rescue_records.append({
                    "model": stem, "language": language,
                    "graded": int(rows["exact_match"].eq(0).sum()),
                    "rescued_from_invalid": int((rows["rescued"] & rows["strict_pred"].eq("[invalid]")).sum()),
                    "rescued_from_other_letter": int((rows["rescued"] & rows["strict_pred"].ne("[invalid]")).sum()),
                })
            rescue_rows.append([display, int(scored["exact_match"].eq(0).sum()),
                                int((scored["rescued"] & scored["strict_pred"].eq("[invalid]")).sum()),
                                int((scored["rescued"] & scored["strict_pred"].ne("[invalid]")).sum())])

        paired = data.pivot(index="question_id_src", columns="language", values="exact_match").sort_index()
        en = paired["en"].to_numpy()
        compact = [display]
        for language in languages[1:]:
            lang = paired[language].to_numpy()
            n = len(en)
            idx = rng.integers(0, n, size=(resamples, n))
            diffs = (lang[idx] - en[idx]).mean(axis=1) * 100
            ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
            gap = (lang.mean() - en.mean()) * 100
            gap_records.append({"model": stem, "language": language, "gap_pp": gap,
                                "ci_low": ci_low, "ci_high": ci_high})
            gap_rows.append([display, language, f"{gap:+.1f}", f"[{ci_low:.1f}, {ci_high:.1f}]"])
            compact.append(f"{gap:+.1f}" + ("*" if ci_low > 0 or ci_high < 0 else ""))
        compact_rows.append(compact)

        subjects = sorted(question_counts)
        subject_accuracies = data.groupby("subject")["exact_match"].mean() * 100
        subject_rows.append([display] + [f"{subject_accuracies[subject]:.1f}" for subject in subjects])
        for subject in subjects:
            subject_records.append({"model": stem, "subject": subject, "accuracy": subject_accuracies[subject],
                                    "n_questions": question_counts[subject]})

        cells = data.groupby(["language", "subject"])["exact_match"].agg(n="size", accuracy="mean")
        subject_pairs = {
            subject: data.loc[data["subject"] == subject].pivot(
                index="question_id_src", columns="language", values="exact_match").sort_index()
            for subject in subjects
        }
        cell_rows, model_gaps = [], []
        for language in languages:
            language_cells = cells.loc[language]
            assert abs(np.average(language_cells["accuracy"], weights=language_cells["n"])
                       - accuracies[language] / 100) <= 1e-12, (
                f"{stem}/{language}: weighted cell accuracy differs from overall accuracy"
            )
            cell_rows.append([language] + [f"{language_cells.loc[s, 'accuracy'] * 100:.1f}" for s in subjects])
            for subject in subjects:
                cell = language_cells.loc[subject]
                n = int(cell["n"])
                assert n == question_counts[subject], f"{stem}/{language}/{subject}: cell question count differs"
                cell_accuracy_records.append({"model": stem, "language": language, "subject": subject,
                                              "n": n, "accuracy": cell["accuracy"] * 100})
                cell_paired = subject_pairs[subject]
                assert len(cell_paired) == n and not cell_paired.isna().any().any(), (
                    f"{stem}/{subject}: subject question_id_src sets differ across languages"
                )
                if language == "en":
                    continue
                en = cell_paired["en"].to_numpy()
                lang = cell_paired[language].to_numpy()
                idx = cell_rng.integers(0, n, size=(resamples, n))
                diffs = (lang[idx] - en[idx]).mean(axis=1) * 100
                ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
                b = int(((en == 1) & (lang == 0)).sum())
                c = int(((en == 0) & (lang == 1)).sum())
                model_gaps.append({
                    "model": stem, "language": language, "subject": subject, "n": n,
                    "en_accuracy": en.mean() * 100, "accuracy": lang.mean() * 100,
                    "gap_pp": (lang.mean() - en.mean()) * 100,
                    "ci_low_pp": ci_low, "ci_high_pp": ci_high,
                    "en_only_correct": b, "lang_only_correct": c, "p_mcnemar": mcnemar(b, c),
                })
        assert len(model_gaps) == 126, f"{stem}: expected 126 language × subject gaps"
        q_values = benjamini_hochberg([row["p_mcnemar"] for row in model_gaps])
        for row, q in zip(model_gaps, q_values):
            assert q >= row["p_mcnemar"], f"{stem}: BH q-value is below its p-value"
            row["q_bh"] = q
        cell_gap_records.extend(model_gaps)
        cell_accuracy_tables.append("\n\n".join([
            f"### {display}",
            markdown_table(["Language"] + [subject.replace("_", " ") for subject in subjects], cell_rows),
        ]))
        significant = sorted((row for row in model_gaps if row["q_bh"] < 0.05), key=lambda row: row["gap_pp"])
        for row in significant:
            cell_gap_rows.append([display, row["language"], row["subject"].replace("_", " "),
                                  f"{row['gap_pp']:+.1f}", f"[{row['ci_low_pp']:.1f}, {row['ci_high_pp']:.1f}]",
                                  f"{row['q_bh']:.3g}"])
        cell_compact_rows.append([display, ", ".join(
            f"{row['language']} · {row['subject'].replace('_', ' ')} {row['gap_pp']:+.1f}"
            for row in significant
        ) if significant else "none"])

    assert included, "No finished main runs found; subject question counts cannot be determined"
    subjects = sorted(question_counts)
    table_a = markdown_table(["Model"] + languages + ["10-lang avg"], accuracy_rows)
    table_b = markdown_table(["Model", "Language", "Gap", "95% CI"], gap_rows)
    table_b_compact = markdown_table(["Model"] + languages[1:], compact_rows)
    table_c = markdown_table(["Model"] + [subject.replace("_", " ") for subject in subjects],
                             subject_rows + [["Questions per language"] + [question_counts[s] for s in subjects]])
    table_d = markdown_table(["Model"] + languages, failure_rows)
    no_lenient = "_No lenient re-scoring yet._"
    table_lenient = markdown_table(["Model"] + languages + ["10-lang avg"], lenient_rows) if lenient_rows else no_lenient
    table_gain = markdown_table(["Model"] + languages, gain_rows) if gain_rows else no_lenient
    table_rescues = markdown_table(
        ["Model", "Strict-wrong graded", "Rescued from [invalid]", "Rescued from another letter"],
        rescue_rows,
    ) if rescue_rows else no_lenient
    table_cell_gaps = markdown_table(["Model", "Language", "Subject", "Gap (pp)", "95% CI", "q"],
                                    cell_gap_rows) if cell_gap_rows else "No cell has q < 0.05."
    table_cell_compact = markdown_table(["Model", "Cells (language · subject, gap)"], cell_compact_rows)
    generated = f"_Generated by `analysis/analyze.py` on {date.today().isoformat()} from {len(included)} of 10 models"
    summary = "\n\n".join([
        "# MMLU-ProX-Lite results summary",
        generated + "._",
        "## Strict accuracy (%) by language",
        table_a,
        "## Language gap vs. English (percentage points, 95% paired bootstrap CI)",
        table_b,
        "## Strict accuracy (%) by subject, pooled over languages",
        table_c,
        "## Answer-extraction failures (`[invalid]`, out of 588 per language)",
        table_d,
        "## Lenient accuracy (%) by language",
        table_lenient,
        "## Lenient minus strict (percentage points)",
        table_gain,
        "## Lenient re-scoring: strict-wrong answers read as correct",
        table_rescues,
        "## Strict accuracy (%) by language and subject",
        "One table per model. Each cell holds that subject's questions in that language (19 to 68).",
        "\n\n".join(cell_accuracy_tables),
        "## Language × subject gaps vs. English (exploratory)",
        "Each cell is the model's accuracy in that language minus its own English accuracy on the same "
        "questions of that subject, in percentage points, with a 95% paired bootstrap CI. p-values are "
        "exact McNemar tests on the paired answers; q-values apply Benjamini–Hochberg correction across "
        "each model's 126 cells. Only cells with q < 0.05 are listed; every cell is in "
        "[`summary/language_subject_gap.csv`](summary/language_subject_gap.csv).",
        table_cell_gaps,
    ]) + "\n"
    # The README leads with the within-model comparisons and leaves out the cross-language average.
    table_a_readme = markdown_table(["Model"] + languages, [row[:-1] for row in accuracy_rows])
    readme_block = "\n" + "\n\n".join([
        "**Language gap vs. English (percentage points).** Each model against its own English "
        "score on the same questions. `*` marks a 95% paired-bootstrap confidence interval that "
        "excludes zero; the intervals are in [`results/summary.md`](results/summary.md).",
        table_b_compact,
        "**Strict accuracy (%) by language**, the scores the gaps above are computed from.",
        table_a_readme,
        "**Strict accuracy (%) by subject, pooled over the 10 languages.** The last row is the "
        "number of questions per subject in each language; small subjects are noisy.",
        table_c,
        "**Language × subject cells that differ from English (exploratory).** A cell is listed when its "
        "gap from the same subject in English survives Benjamini–Hochberg correction (q < 0.05) across "
        "that model's 126 cells, using exact McNemar tests. Gaps are in percentage points; every cell is "
        "in [`results/summary/language_subject_gap.csv`](results/summary/language_subject_gap.csv).",
        table_cell_compact,
        "**Lenient re-scoring: percentage points gained over strict** (method under "
        "[Methodology](#methodology-protocol-v1)). Large gains would mean answers given in a format "
        "the strict regex misses.",
        table_gain if gain_rows else "_Lenient re-scoring has not run yet._",
    ]) + "\n\n"

    output_dir = ROOT / "results/summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in (
        ("accuracy_by_language.csv", accuracy_records),
        ("language_gap.csv", gap_records),
        ("accuracy_by_subject.csv", subject_records),
        ("extraction_failures.csv", failure_records),
        ("accuracy_by_language_subject.csv", cell_accuracy_records),
        ("language_subject_gap.csv", cell_gap_records),
    ):
        pd.DataFrame(records).to_csv(output_dir / filename, index=False)
    pd.DataFrame(lenient_records, columns=["model", "language", "strict", "lenient", "gain_pp"]).to_csv(
        output_dir / "lenient_accuracy_by_language.csv", index=False)
    pd.DataFrame(rescue_records, columns=["model", "language", "graded", "rescued_from_invalid",
                                         "rescued_from_other_letter"]).to_csv(
        output_dir / "lenient_rescues.csv", index=False)
    (ROOT / "results/summary.md").write_text(summary, encoding="utf-8")
    args.readme.write_bytes(prefix + readme_block.encode("utf-8") + suffix)
    print(f"Included {len(included)} of 10 models: {', '.join(included)}")
    print("All self-checks passed (140 files, 5880 rows, paired IDs, and harness accuracies per model).")
    print(f"Bootstrap: {resamples} resamples per gap; seed 20261012.")
    print("Wrote results/summary.md and 8 CSVs in results/summary/.")
    print(f"Updated results block in {args.readme}.")
    print(f"Elapsed: {perf_counter() - started:.2f} seconds.")


if __name__ == "__main__":
    main()
