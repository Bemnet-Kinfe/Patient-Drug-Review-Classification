"""Classifier 1: TF-IDF + Linear SVM (One-vs-Rest and One-vs-One).

Usage (from the repo root):
    python 01_svm_tfidf/svm_tfidf.py
    python 01_svm_tfidf/svm_tfidf.py --train path/to/train.tsv --test path/to/test.tsv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC

from common.data_utils import TOP_CONDITIONS, ensure_results_dir, load_data


def build_features(train_text, test_text, max_features=30000):
    """Unigram + bigram TF-IDF, English stop words removed, vocabulary capped at 30k."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
    )
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    print(f"TF-IDF matrix: train {X_train.shape}, test {X_test.shape}")
    return vectorizer, X_train, X_test


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", help="Path to training file (.csv or .tsv)")
    parser.add_argument("--test", help="Path to test file (.csv or .tsv)")
    parser.add_argument("--C", type=float, default=1.0, help="LinearSVC regularization (default 1.0)")
    args = parser.parse_args()

    train, test = load_data(args.train, args.test)
    _, X_train, X_test = build_features(train["review"], test["review"])
    y_train, y_test = train["condition"], test["condition"]

    base = LinearSVC(C=args.C, class_weight="balanced")
    models = {
        "One-vs-Rest": OneVsRestClassifier(base),
        "One-vs-One": OneVsOneClassifier(base),  # 6 classes -> 15 pairwise classifiers
    }

    out_dir = ensure_results_dir("svm_tfidf")
    summary = []

    for name, model in models.items():
        print(f"\n=== SVM + TF-IDF ({name}) ===")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        summary.append({
            "model": f"SVM + TF-IDF ({name})",
            "accuracy": accuracy_score(y_test, preds),
            "macro_f1": f1_score(y_test, preds, average="macro"),
            "weighted_f1": f1_score(y_test, preds, average="weighted"),
        })
        print(classification_report(y_test, preds, digits=4))

        cm = pd.DataFrame(
            confusion_matrix(y_test, preds, labels=TOP_CONDITIONS),
            index=TOP_CONDITIONS, columns=TOP_CONDITIONS,
        )
        print("Confusion matrix (rows = true, cols = predicted):")
        print(cm)

        tag = name.lower().replace("-", "")
        cm.to_csv(out_dir / f"confusion_matrix_{tag}.csv")
        pd.DataFrame({"condition": y_test, "pred": preds}).to_csv(
            out_dir / f"predictions_{tag}.csv", index=False
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== Summary ===")
    print(summary_df.round(4).to_string(index=False))
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
