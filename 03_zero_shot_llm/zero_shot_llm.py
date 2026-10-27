"""Classifier 3: zero-shot LLM classification with the Anthropic Claude API.

Compares two prompt variants (minimal vs. descriptive) on a stratified sample
of the test set (default: 500 reviews per condition = 3,000 reviews).

Requires the ANTHROPIC_API_KEY environment variable.

Usage (from the repo root):
    python 03_zero_shot_llm/zero_shot_llm.py --sanity-check        # 5 reviews, quick test
    python 03_zero_shot_llm/zero_shot_llm.py                       # full run (~6,000 API calls)
    python 03_zero_shot_llm/zero_shot_llm.py --n-per-class 50      # cheaper run
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anthropic
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

from common.data_utils import TOP_CONDITIONS, ensure_results_dir, load_data

MODEL = "claude-haiku-4-5-20251001"  # model used for the reported results
CONDITIONS = TOP_CONDITIONS


def prompt_a(review: str) -> str:
    """Prompt A (minimal)."""
    conditions_str = ", ".join(CONDITIONS)
    return f"""Classify the following patient drug review into exactly one of these medical conditions: {conditions_str}

Review: {review}

Respond with only the condition name, nothing else."""


def prompt_b(review: str) -> str:
    """Prompt B (descriptive, with expert framing and guidelines)."""
    conditions_str = "\n".join(f"- {c}" for c in CONDITIONS)
    return f"""You are an expert medical text classifier specializing in analyzing patient-reported drug reviews. Your task is to identify the medical condition a patient is being treated for, based solely on the language, symptoms, and context described in their review.

Guidelines:
- Focus on symptom descriptions, drug names, and treatment outcomes mentioned
- Do not rely on the drug name alone — use the full context of the review
- If the review mentions multiple conditions, select the condition that the drug being reviewed is primarily prescribed to treat

Choose exactly one condition from the following list:
{conditions_str}

Patient review: {review}

Respond with only the condition name from the list above. Do not include any explanation, punctuation, or additional text."""


PROMPTS = {"a": prompt_a, "b": prompt_b}


def parse_label(raw: str) -> str:
    """Map the model's reply to one of the six conditions (substring match), else 'unknown'.

    Note: matching is on the exact condition strings, so a reply like
    "Diabetes Type 2" (no comma) does not match "Diabetes, Type 2" and is
    counted as 'unknown'. This is the behaviour behind the reported results.
    """
    raw = raw.strip().lower()
    for condition in CONDITIONS:
        if condition.lower() in raw:
            return condition
    return "unknown"


def classify(client, prompt_text: str, delay: float = 0.3, retries: int = 3) -> tuple[str, str]:
    """Return (parsed_label, raw_reply)."""
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=20,
                messages=[{"role": "user", "content": prompt_text}],
            )
            raw = response.content[0].text
            time.sleep(delay)
            return parse_label(raw), raw.strip()
        except anthropic.APIError as e:
            wait = 2 ** attempt
            print(f"API error ({e.__class__.__name__}), retrying in {wait}s...")
            time.sleep(wait)
    return "unknown", ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", help="Path to training file (.csv or .tsv)")
    parser.add_argument("--test", help="Path to test file (.csv or .tsv)")
    parser.add_argument("--n-per-class", type=int, default=500, help="Reviews sampled per condition (default 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts", default="ab", help="Which prompts to run: 'a', 'b' or 'ab' (default)")
    parser.add_argument("--sanity-check", action="store_true", help="Classify 5 test reviews with Prompt A and exit")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set the ANTHROPIC_API_KEY environment variable first (see 03_zero_shot_llm/README.md).")
    client = anthropic.Anthropic()

    _, test = load_data(args.train, args.test, verbose=False)

    if args.sanity_check:
        print("=== Sanity check on 5 reviews (Prompt A) ===")
        for _, row in test.head(5).iterrows():
            pred, raw = classify(client, prompt_a(row["review"]))
            print(f"True: {row['condition']:<18} | Pred: {pred:<18} | Raw: {raw!r}")
        return

    # Stratified sample: n reviews per condition
    sample = (
        test.groupby("condition")
        .sample(n=args.n_per_class, random_state=args.seed)
        .reset_index(drop=True)
    )
    print(f"Sample size: {len(sample):,} ({args.n_per_class} per condition)")

    out_dir = ensure_results_dir("zero_shot_llm")

    for key in args.prompts.lower():
        prompt_fn = PROMPTS[key]
        print(f"\nRunning Prompt {key.upper()}...")
        results = [classify(client, prompt_fn(r)) for r in tqdm(sample["review"], desc=f"Prompt {key.upper()}")]
        sample[f"pred_{key}"] = [r[0] for r in results]
        sample[f"raw_{key}"] = [r[1] for r in results]

        print(f"\n=== Prompt {key.upper()} results ===")
        print(classification_report(sample["condition"], sample[f"pred_{key}"], digits=4, zero_division=0))
        n_unknown = (sample[f"pred_{key}"] == "unknown").sum()
        print(f"Unknown predictions: {n_unknown}")
        if n_unknown:
            print("Unknowns by true condition:")
            print(sample.loc[sample[f"pred_{key}"] == "unknown", "condition"].value_counts().to_string())

        cm = pd.DataFrame(
            confusion_matrix(sample["condition"], sample[f"pred_{key}"], labels=CONDITIONS),
            index=CONDITIONS, columns=CONDITIONS,
        )
        cm.to_csv(out_dir / f"confusion_matrix_prompt_{key}.csv")

    sample.to_csv(out_dir / "zero_shot_results.csv", index=False)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
