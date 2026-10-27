"""Shared data loading and filtering used by all three classifiers."""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

# The six target conditions (strings match the UCI dataset's `condition` column exactly).
TOP_CONDITIONS = [
    "Birth Control",
    "Depression",
    "Anxiety",
    "Acne",
    "ADHD",
    "Diabetes, Type 2",
]

# Default file names. The UCI download ships as .tsv; a preprocessed .csv works too.
DEFAULT_TRAIN_CANDIDATES = ["drugsComTrain_labeled.csv", "drugsComTrain_raw.tsv", "drugsComTrain_raw.csv"]
DEFAULT_TEST_CANDIDATES = ["drugsComTest_labeled.csv", "drugsComTest_raw.tsv", "drugsComTest_raw.csv"]


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def _resolve(path, candidates):
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")
        return p
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"None of {candidates} found in {DATA_DIR}. "
        "See data/README.md for download instructions, or pass --train/--test paths."
    )


def load_data(train_path=None, test_path=None, conditions=TOP_CONDITIONS, verbose=True):
    """Load the train/test splits and keep only reviews for the target conditions.

    Returns (train_filtered, test_filtered) with a reset index.
    """
    train_df = _read_table(_resolve(train_path, DEFAULT_TRAIN_CANDIDATES))
    test_df = _read_table(_resolve(test_path, DEFAULT_TEST_CANDIDATES))

    # Reviews with a missing condition label cannot be used for supervised learning.
    train_df = train_df.dropna(subset=["condition", "review"])
    test_df = test_df.dropna(subset=["condition", "review"])

    train = train_df[train_df["condition"].isin(conditions)].reset_index(drop=True)
    test = test_df[test_df["condition"].isin(conditions)].reset_index(drop=True)

    if verbose:
        print(f"Train: {len(train_df):,} rows -> {len(train):,} after filtering")
        print(f"Test:  {len(test_df):,} rows -> {len(test):,} after filtering")
        print("\nTrain class distribution:")
        print(train["condition"].value_counts().to_string())

    return train, test


def ensure_results_dir(subdir: str) -> Path:
    out = RESULTS_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out
