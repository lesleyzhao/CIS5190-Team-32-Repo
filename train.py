"""
Run improved models using this command:
    python train.py

current model: DistillBERT
"""

from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from DistillBERT import Model, DEVICE          # DistilBERT model
from preprocess import prepare_data      # loads + cleans the CSV


CSV_PATH    = Path("scraped_headlines_clean_latest.csv")
WEIGHTS_OUT = Path("model_weights.pt")

BATCH_SIZE  = 32
EPOCHS      = 5
LR_ENCODER  = 2e-5     # small LR for pretrained encoder
LR_HEAD     = 2e-4     # larger LR for the new classifier head
WEIGHT_DECAY= 1e-2
TEST_SIZE   = 0.20
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# dataset wrapper 
class HeadlineDataset(Dataset):
    def __init__(self, texts: list[str], labels: torch.Tensor) -> None:
        self.texts  = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.labels[idx]


def make_collate(model: Model):
    """Tokenize a batch of (text, label) pairs using the model's tokenizer."""
    def collate(batch):
        texts, labels = zip(*batch)
        enc = model._tokenize(list(texts))
        return enc["input_ids"], enc["attention_mask"], torch.stack(list(labels))
    return collate


# training
def main() -> None:
    # load data using preprocess.py
    print(f"Loading: {CSV_PATH}")
    X, y = prepare_data(str(CSV_PATH))          # List[str], LongTensor
    print(f"Total rows: {len(X)}  |  FoxNews: {y.sum().item()}  |  NBC: {(y==0).sum().item()}\n")

    # train / val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y.tolist(), test_size=TEST_SIZE, random_state=SEED, stratify=y.tolist()
    )
    y_tr  = torch.tensor(y_tr,  dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    print(f"Train: {len(X_tr)}  |  Val: {len(X_val)}\n")

    # build model
    model = Model()
    collate = make_collate(model)

    # dataLoaders
    train_loader = DataLoader(HeadlineDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE,   shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(HeadlineDataset(X_val, y_val), batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=collate)

    # optimizer — lower LR for encoder, higher for head
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),    "lr": LR_ENCODER},
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=len(train_loader) * EPOCHS
    )

    criterion  = nn.CrossEntropyLoss()
    best_acc   = 0.0

    # training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for input_ids, attn_mask, labels in train_loader:
            input_ids, attn_mask, labels = (
                input_ids.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)
            )
            optimizer.zero_grad()
            loss = criterion(model(input_ids, attn_mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # validate 
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for input_ids, attn_mask, labels in val_loader:
                logits = model(input_ids.to(DEVICE), attn_mask.to(DEVICE))
                preds_all.extend(logits.argmax(-1).cpu().tolist())
                labels_all.extend(labels.tolist())

        val_acc = accuracy_score(labels_all, preds_all)
        print(f"Epoch {epoch}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")

        # save best checkpoint 
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_OUT)
            print(f"Saved best model to {WEIGHTS_OUT}  (acc={best_acc:.4f})")

    # final report
    print(f"\nBest val accuracy: {best_acc:.4f}")
    model.load_state_dict(torch.load(WEIGHTS_OUT, map_location=DEVICE))
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for input_ids, attn_mask, labels in val_loader:
            logits = model(input_ids.to(DEVICE), attn_mask.to(DEVICE))
            preds_all.extend(logits.argmax(-1).cpu().tolist())
            labels_all.extend(labels.tolist())

    print("\n── Classification Report ─────────────────────────────────────────")
    print(classification_report(labels_all, preds_all, target_names=["NBC (0)", "FoxNews (1)"]))
    print(f"\nDone. Submit these two files to the leaderboard:")
    print(f"  • model.py")
    print(f"  • {WEIGHTS_OUT}")


if __name__ == "__main__":
    main()