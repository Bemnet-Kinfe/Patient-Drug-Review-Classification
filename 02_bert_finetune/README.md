# Classifier 2: Fine-Tuned BERT

Fine-tunes `bert-base-uncased` end-to-end for 6-way condition classification.

Unlike TF-IDF, BERT reads each word in context. That means it can tell *"not anxious"* apart from *"anxious"*. We expected this to help most on the Anxiety/Depression boundary.

## Method

| Setting | Value |
|---------|-------|
| Base model | `bert-base-uncased` (12 layers, 110M params) + linear classification head |
| Max sequence length | 128 tokens |
| Batch size | 32 |
| Epochs | 3 |
| Optimizer | AdamW, lr = 2e-5, weight decay = 0.01 |
| Schedule | Linear warmup over first 10% of steps, then linear decay |
| Gradient clipping | max norm 1.0 |
| Class imbalance | Inverse-frequency class weights in the cross-entropy loss (0.32 for Birth Control up to 3.61 for Diabetes, Type 2) |
| Hardware | Single NVIDIA T4 GPU, about 25 minutes |

## Run

From the repo root:

```bash
python 02_bert_finetune/bert_finetune.py
# options
python 02_bert_finetune/bert_finetune.py --epochs 3 --batch-size 32 --max-len 128 --save-dir models/bert_finetuned
```

A GPU is strongly recommended. To use Google Colab:
1. Clone the repo.
2. Install the requirements.
3. Put the data in `data/`.
4. Run the script from the repo root.

Outputs:
- `results/bert_finetune/predictions.csv` and `confusion_matrix.csv`
- The fine-tuned model and `label_classes.json`, saved to `models/bert_finetuned/`. This folder is git-ignored because the weights are about 440 MB.

## Results (test set, n = 18,432)

Training was stable. Loss fell 0.555 → 0.184 → 0.101 over the three epochs, and training accuracy rose 81.3% → 94.6% → 97.3%.

| Condition | Precision | Recall | F1 |
|-----------|----------:|-------:|---:|
| ADHD | 0.96 | 0.95 | 0.96 |
| Acne | 0.93 | 0.97 | 0.95 |
| Anxiety | 0.88 | 0.89 | 0.88 |
| Birth Control | 0.99 | 0.98 | 0.99 |
| Depression | 0.91 | 0.92 | 0.92 |
| Diabetes, Type 2 | 0.97 | 0.97 | 0.97 |
| **Macro avg** | **0.94** | **0.95** | **0.94** |

BERT matches the SVM within about 1 point. Its contextual embeddings only slightly improved Anxiety recall (0.89 vs. 0.86 for the SVM). This suggests the overlap between Anxiety and Depression is too large for context alone to resolve.
