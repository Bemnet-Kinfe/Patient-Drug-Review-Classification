"""Classifier 2: fine-tuned bert-base-uncased with class-weighted cross-entropy.

Usage (from the repo root; a GPU is strongly recommended):
    python 02_bert_finetune/bert_finetune.py
    python 02_bert_finetune/bert_finetune.py --epochs 3 --save-dir models/bert_finetuned
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

from common.data_utils import REPO_ROOT, TOP_CONDITIONS, ensure_results_dir, load_data

MODEL_NAME = "bert-base-uncased"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DrugReviewDataset(Dataset):
    """Tokenizes drug reviews on the fly."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # standard for transformers
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        trues.extend(batch["labels"].numpy())
    return np.array(preds), np.array(trues)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", help="Path to training file (.csv or .tsv)")
    parser.add_argument("--test", help="Path to test file (.csv or .tsv)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default=str(REPO_ROOT / "models" / "bert_finetuned"),
                        help="Where to save the fine-tuned model (git-ignored by default)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train, test = load_data(args.train, args.test)

    # Encode labels
    le = LabelEncoder().fit(TOP_CONDITIONS)
    y_train = le.transform(train["condition"])
    y_test = le.transform(test["condition"])
    print("\nLabel mapping:", dict(enumerate(le.classes_)))

    # Inverse-frequency class weights (mirrors class_weight='balanced' in the SVM)
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    print("Class weights:", dict(zip(le.classes_, class_weights.round(3))))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_ds = DrugReviewDataset(train["review"].tolist(), y_train, tokenizer, args.max_len)
    test_ds = DrugReviewDataset(test["review"].tolist(), y_test, tokenizer, args.max_len)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(le.classes_)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs} | loss {loss:.4f} | train acc {acc:.4f}")

    preds, trues = evaluate(model, test_loader, device)
    print("\n=== BERT Classification Report (test set) ===")
    print(classification_report(trues, preds, target_names=le.classes_, digits=4))

    # Save predictions + confusion matrix
    out_dir = ensure_results_dir("bert_finetune")
    pd.DataFrame({
        "condition": le.inverse_transform(trues),
        "pred": le.inverse_transform(preds),
    }).to_csv(out_dir / "predictions.csv", index=False)
    cm = pd.DataFrame(
        confusion_matrix(le.inverse_transform(trues), le.inverse_transform(preds), labels=TOP_CONDITIONS),
        index=TOP_CONDITIONS, columns=TOP_CONDITIONS,
    )
    cm.to_csv(out_dir / "confusion_matrix.csv")
    print(f"Results saved to {out_dir}")

    # Save the model so it doesn't need re-training
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(save_dir / "label_classes.json", "w") as f:
        json.dump(le.classes_.tolist(), f)
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
