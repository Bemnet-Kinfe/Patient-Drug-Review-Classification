# Patient Drug Review Classification

**A multi-classifier NLP comparison for identifying the treated condition from patient-written drug reviews.**

CBB7500 Final Project — Group 13: Christopher Alvarez, Lydia Bao, Bemnet Nigatu Kinfe, Dongnan Liu

---

## Overview

Patient drug reviews are usually mined for sentiment or side effects. This project asks a different question: **can we tell which condition a patient is being treated for, from the review text alone?**

We benchmark three classifier families on the same data and evaluation protocol:

| # | Method | Paradigm | Folder |
|---|--------|----------|--------|
| 1 | TF-IDF + Linear SVM (OvR & OvO) | Classical sparse features | [`01_svm_tfidf/`](01_svm_tfidf/) |
| 2 | Fine-tuned bert-base-uncased | Supervised deep learning | [`02_bert_finetune/`](02_bert_finetune/) |
| 3 | Zero-shot Claude (2 prompt variants) | Prompt-only, no training | [`03_zero_shot_llm/`](03_zero_shot_llm/) |

**Target conditions (6):** Birth Control, Depression, Anxiety, Acne, ADHD, Diabetes Type 2.

Only the free-text review is used as model input. Drug name and rating fields are excluded.

## Results

| Model | Eval set | Accuracy | Macro F1 |
|-------|----------|---------:|---------:|
| SVM + TF-IDF (One-vs-Rest) | Full test set (n = 18,432) | 0.962 | **0.946** |
| SVM + TF-IDF (One-vs-One) | Full test set (n = 18,432) | 0.961 | 0.946 |
| Fine-tuned BERT | Full test set (n = 18,432) | — | 0.94 |
| Zero-shot LLM, Prompt A (minimal) | Stratified sample (n = 3,000) | 0.77 | 0.77 |
| Zero-shot LLM, Prompt B (descriptive) | Stratified sample (n = 3,000) | 0.90 | 0.91 |

### Key findings

- **Classical methods remain strong.** TF-IDF + SVM matched or beat BERT, trains in seconds, and is interpretable.
- **Anxiety vs. Depression is the hardest boundary** for every model (F1 ≈ 0.84–0.92). The difficulty seems to come from the data itself: the two conditions share drugs, symptoms and vocabulary.
- **Distinct vocabulary beats class size.** Diabetes Type 2 is the smallest class but scores F1 ≥ 0.96 across models thanks to terms like *metformin*, *A1C* and *glucose*.
- **Prompt design is a major lever.** Rewording the zero-shot prompt raised macro F1 by 14 points, more than the gap between SVM and BERT.

Full write-up: [`report/Group13_Project_Report.pdf`](report/Group13_Project_Report.pdf)

## Repository structure

```
.
├── 01_svm_tfidf/          # Classifier 1: TF-IDF + LinearSVC
├── 02_bert_finetune/      # Classifier 2: fine-tuned BERT
├── 03_zero_shot_llm/      # Classifier 3: zero-shot Claude API
├── common_data_utils.py   # Shared loading + filtering to the 6 conditions
├── report/                # Final project report (PDF)
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/<your-username>/drug-review-condition-classification.git
cd drug-review-condition-classification
pip install -r requirements.txt
```

1. **Get the data.** Download the [UCI Drug Review Dataset (Drugs.com)](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com) and place the train and test files in `data/`. See [`data/README.md`](data/README.md).
2. **Run any classifier from the repo root:**

```bash
python 01_svm_tfidf/svm_tfidf.py
python 02_bert_finetune/bert_finetune.py        # GPU recommended
export ANTHROPIC_API_KEY=your_key_here
python 03_zero_shot_llm/zero_shot_llm.py
```

Each folder's README explains its method, settings and options.

## Data and preprocessing

- **Source:** UCI Drug Review Dataset. It contains 215,063 reviews covering 3,654 drugs and 916 conditions, and comes pre-split into train and test sets (about 3:1).
- **Filtering:** Reviews with no condition label are dropped, and the rest are filtered to the six target conditions. That leaves **55,286 training** and **18,432 test** reviews.
- **Class imbalance:** Birth Control has 28,788 training reviews and Diabetes Type 2 has 2,554, an 11:1 ratio. We use **class weighting** instead of downsampling, so no training data is thrown away. Results are reported with **macro F1** so that every condition counts equally.

### Why these six conditions?

- Each has at least 2,500 training reviews.
- They span distinct clinical domains: reproductive, mental health, dermatology, neurodevelopmental and metabolic.
- Each has fairly separable vocabulary.
- **Pain** was excluded as too vague.
- **Anxiety and Stress** was excluded as a near-duplicate of Anxiety.
- **Bipolar Disorder, Insomnia, Weight Loss and Obesity** were excluded for overlapping vocabulary with the chosen conditions or weak separation signal.

## Citation

Dataset: Kallumadi, S., & Grässer, F. (2018). *Drug Review Dataset (Drugs.com)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5SK5S

See the report for the full reference list.
