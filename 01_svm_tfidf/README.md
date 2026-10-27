# Classifier 1: TF-IDF + Linear SVM

A classical baseline that pairs sparse TF-IDF features with a linear Support Vector Machine.

## Method

| Setting | Value |
|---------|-------|
| Features | TF-IDF, unigrams + bigrams (`ngram_range=(1, 2)`) |
| Vocabulary | Top 30,000 terms, English stop words removed, lowercased |
| Model | `LinearSVC(C=1.0, class_weight="balanced")` |
| Multiclass | One-vs-Rest (6 classifiers) **and** One-vs-One (15 pairwise classifiers) |
| Eval set | Full held-out test set (n = 18,432) |

Using bigrams lets the model pick up compound phrases such as *"birth control"* and *"panic attack"*. Setting `class_weight="balanced"` makes mistakes on smaller classes cost more, which counters the 11:1 imbalance between the largest and smallest classes.

## Run

From the repo root:

```bash
python 01_svm_tfidf/svm_tfidf.py
# or with explicit paths
python 01_svm_tfidf/svm_tfidf.py --train data/drugsComTrain_raw.tsv --test data/drugsComTest_raw.tsv
```

Training takes seconds on a CPU. Outputs are written to `results/svm_tfidf/`:
- `summary.csv`
- `predictions_*.csv`
- `confusion_matrix_*.csv`

## Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|---------:|---------:|------------:|
| One-vs-Rest | 0.9615 | 0.9458 | 0.9614 |
| One-vs-One | 0.9614 | 0.9458 | 0.9614 |

OvR and OvO are effectively identical, so the choice of decomposition does not matter for this task.

**Per-condition F1 (OvR):**

| Condition | F1 |
|-----------|---:|
| Birth Control | 0.99 |
| Acne | 0.96 |
| ADHD | 0.96 |
| Diabetes, Type 2 | 0.96 |
| Depression | 0.92 |
| Anxiety | 0.88 |

**Main error patterns:**
- **Anxiety → Depression** (230 errors, about 12% of Anxiety reviews). The two conditions share drugs and symptom language.
- **Acne → Birth Control** (59 errors). Oral contraceptives are often prescribed for acne, so these reviews talk about birth control.
