"""Build results/recovered-usage/summary.md and the README token tables from the recovered usage.

    uv run python analysis/usage_recovery/summarize.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze import ORDER, REFERENCE, markdown_table, readme_sections  # noqa: E402

DIR = ROOT / "results/recovered-usage"
LANGUAGES = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())["languages"]

HEADER = """# Recovered token usage of the main run

The main run kept only each answer's text. This table set restores per-question token usage from
the providers' own records (Neuralwatt's usage API, LLM Gateway's request logs), which carry no
question text and were tied to questions after the fact; see [README](../../README.md#how-the-usage-was-recovered)
for the method. Counts are each provider's, in each model's own tokenizer: compare languages within
a model, not counts across models.

- Reasoning tokens are **reported** by LLM Gateway, and **estimated** for the Neuralwatt models as
  output tokens − answer tokens − a fixed overhead (2; Gemma 5), which matched the reported or
  tokenized reasoning within ±1 token on re-run requests. A missing gateway value counts as 0.
- Medians and ratios use only matched questions.
"""


def by_language(frame, column):
    """Median per question in each language; total in each language ÷ total on the same questions in English."""
    medians = frame.groupby("language")[column].median()
    paired = frame.pivot(index="question_id_src", columns="language", values=column)
    ratios = []
    for lang in LANGUAGES[1:]:
        both = paired[["en", lang]].dropna()
        ratios.append(f"{both[lang].sum() / both['en'].sum():.2f}" if both["en"].sum() else "–")
    return [f"{medians[lang]:,.0f}" for lang in LANGUAGES], ratios


def main():
    nw = pd.read_csv(DIR / "neuralwatt.csv").rename(columns={"reasoning_tokens_est": "reasoning_tokens"})
    gw = pd.read_csv(DIR / "llmgateway.csv")
    gw.loc[gw["log_id"].notna(), "reasoning_tokens"] = gw["reasoning_tokens"].fillna(0)
    data = pd.concat([gw.assign(source="reported"), nw.assign(source="estimated")], ignore_index=True)
    full, data = data, data[data["completion_tokens"].notna()]
    checks = json.loads((DIR / "checks.json").read_text())
    names = {stem: yaml.safe_load((ROOT / "configs/models" / f"{stem}.yaml").read_text())["name"]
             + (" †" if stem in REFERENCE else "") for stem in ORDER}

    coverage, medians, ratios, output_medians, output_ratios, right_wrong, caps = [], [], [], [], [], [], []
    for stem in ORDER:
        frame, check = data[data["model"] == stem], checks[stem]
        name = names[stem]
        verified = f"{check['right']}/{check['checked']} ({check['check']})" if "check" in check else "exact"
        coverage.append([name, f"{check['matched']:,}", check["method"], verified, frame["source"].iloc[0],
                         f"{frame['reasoning_tokens'].gt(0).mean():.0%}"])
        for column, med, rat in (("reasoning_tokens", medians, ratios), ("completion_tokens", output_medians, output_ratios)):
            m, r = by_language(frame, column)
            med.append([name] + m)
            rat.append([name] + r)
        by_correct = frame.groupby("correct")["reasoning_tokens"].median()
        right_wrong.append([name, f"{by_correct.get(1, float('nan')):,.0f}", f"{by_correct.get(0, float('nan')):,.0f}"])
        empty = full[(full["model"] == stem) & (full["empty_answer"] == 1)]
        caps.append([name, int(frame["finish_reason"].eq("length").sum()), len(empty),
                     ", ".join(f"{reason} {n}" for reason, n in empty["finish_reason"].fillna("no record").value_counts().items()) or "–"])

    sections = [
        HEADER,
        "## Coverage\n",
        "Matched: questions tied to a usage record (of 5,880). Check: on questions whose record is known"
        " independently (re-run requests, or anchors held out of the alignment), how many were"
        " tied to the right record; misses were swaps between neighbouring questions. GPT-6 Luna and"
        " GPT-6.1 Sol miss the 68 Swahili math questions blocked by the content filter. Any reasoning:"
        " share of matched questions with more than 0 reasoning tokens.\n",
        markdown_table(["Model", "Matched", "Method", "Check", "Reasoning tokens", "Any reasoning"], coverage),
        "\n## Reasoning tokens by language\n",
        "**Median reasoning tokens per question:**\n",
        markdown_table(["Model"] + LANGUAGES, medians),
        "\n**Relative to English**: total reasoning tokens in the language ÷ total on the same questions in"
        " English. Part of a difference can be tokenization (some scripts take more tokens) or the"
        " language the model reasons in, not only more reasoning.\n",
        markdown_table(["Model"] + LANGUAGES[1:], ratios),
        "\n## Output tokens by language\n",
        "All output tokens (reasoning + answer, `completion_tokens`). **Median per question:**\n",
        markdown_table(["Model"] + LANGUAGES, output_medians),
        "\n**Relative to English** (as above):\n",
        markdown_table(["Model"] + LANGUAGES[1:], output_ratios),
        "\n## Right and wrong answers\n",
        "Median reasoning tokens per question, all languages:\n",
        markdown_table(["Model", "Right", "Wrong"], right_wrong),
        "\n## Token cap and empty answers\n",
        "Hit cap: `finish_reason = length` (the output reached the model's max tokens). Empty answers: as"
        " logged by lm-eval, by the finish reason of their record (no record: the blocked Swahili math"
        " questions of the GPT models).\n",
        markdown_table(["Model", "Hit cap", "Empty answers", "Finish reason of empty answers"], caps),
        "",
    ]
    (DIR / "summary.md").write_text("\n".join(sections), encoding="utf-8")
    print("wrote", DIR / "summary.md")

    readme = ROOT / "README.md"
    prefix, suffix = readme_sections(readme, "usage")
    relative = lambda med, rat: [[m[0], m[1]] + r[1:] for m, r in zip(med, rat)]  # English median, then ratios
    header = ["Model", "en (tokens)"] + [f"{lang} ×" for lang in LANGUAGES[1:]]
    block = "\n".join([
        "",
        "**Output tokens (reasoning + answer) relative to English.** en: median tokens per English question."
        " Other columns: total tokens in that language ÷ total on the same questions in English, so 1.30 means"
        " 30% more. Matched questions only.",
        "",
        markdown_table(header, relative(output_medians, output_ratios)),
        "",
        "**Reasoning tokens relative to English**, same layout. Reported by LLM Gateway; estimated for the"
        " Neuralwatt models. Claude Sonnet 5.5 and GPT-6.1 Sol answer most questions without reasoning"
        " (English median 0), so their ratios rest on the questions where they did reason.",
        "",
        markdown_table(header, relative(medians, ratios)),
        "",
    ])
    readme.write_bytes(prefix + block.encode("utf-8") + suffix)
    print("wrote token tables to", readme)


if __name__ == "__main__":
    main()
