import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class Config:
    manifest_path: str = "manifest.csv"
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 7
    random_seed: int = 42

    # Expected input shape per sample: (time_steps, num_subcarriers)
    time_steps: int = 120
    num_subcarriers: int = 64

    # Model hyperparameters
    conv_channels_1: int = 32
    conv_channels_2: int = 64
    gru_hidden: int = 128
    gru_layers: int = 1
    dropout: float = 0.3

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "best_cnn_gru.pt"


CFG = Config()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zscore_per_sample(x: np.ndarray) -> np.ndarray:
    mean = x.mean()
    std = x.std()
    if std < 1e-6:
        std = 1.0
    return (x - mean) / std


class CSIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, time_steps: int, num_subcarriers: int):
        self.df = df.reset_index(drop=True)
        self.time_steps = time_steps
        self.num_subcarriers = num_subcarriers

    def __len__(self):
        return len(self.df)

    def _fix_shape(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={x.shape}")

        t, s = x.shape

        if t > self.time_steps:
            x = x[:self.time_steps, :]
        elif t < self.time_steps:
            pad = np.zeros((self.time_steps - t, s), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=0)

        if s > self.num_subcarriers:
            x = x[:, :self.num_subcarriers]
        elif s < self.num_subcarriers:
            pad = np.zeros((self.time_steps, self.num_subcarriers - s), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=1)

        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        label = int(row["label"])

        x = np.load(path).astype(np.float32)  # expected shape: (T, S)
        x = self._fix_shape(x)
        x = zscore_per_sample(x)

        # CNN input shape: (C, T, S), where C=1
        x = np.expand_dims(x, axis=0)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class CNNGRUClassifier(nn.Module):
    def __init__(
        self,
        input_time_steps: int,
        input_subcarriers: int,
        conv_channels_1: int = 32,
        conv_channels_2: int = 64,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, conv_channels_1, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(conv_channels_1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # reduce time axis only

            nn.Conv2d(conv_channels_1, conv_channels_2, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(conv_channels_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        reduced_time = input_time_steps // 4
        if reduced_time < 1:
            raise ValueError("time_steps is too small after pooling.")

        gru_input_size = conv_channels_2 * input_subcarriers

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, T, S)
        x = self.feature_extractor(x)  # (B, C, T', S)
        b, c, t, s = x.shape

        # GRU input: (B, T', C*S)
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * s)

        out, _ = self.gru(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        logits = self.classifier(last)
        return logits


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)

        all_labels.extend(y.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm,
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    return total_loss / total_count


def main():
    seed_everything(CFG.random_seed)
    device = CFG.device
    print(f"device: {device}")

    df = pd.read_csv(CFG.manifest_path)

    missing = [p for p in df["path"].tolist() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing files example: {missing[:5]}")

    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=CFG.random_seed,
        stratify=df["label"],
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        random_state=CFG.random_seed,
        stratify=train_df["label"],
    )

    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_dataset = CSIDataset(train_df, CFG.time_steps, CFG.num_subcarriers)
    val_dataset = CSIDataset(val_df, CFG.time_steps, CFG.num_subcarriers)
    test_dataset = CSIDataset(test_df, CFG.time_steps, CFG.num_subcarriers)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
    )

    model = CNNGRUClassifier(
        input_time_steps=CFG.time_steps,
        input_subcarriers=CFG.num_subcarriers,
        conv_channels_1=CFG.conv_channels_1,
        conv_channels_2=CFG.conv_channels_2,
        gru_hidden=CFG.gru_hidden,
        gru_layers=CFG.gru_layers,
        dropout=CFG.dropout,
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), CFG.save_path)
            print(f"  -> best model saved to {CFG.save_path}")
        else:
            patience_counter += 1
            print(f"  -> no improvement ({patience_counter}/{CFG.patience})")

        if patience_counter >= CFG.patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(CFG.save_path, map_location=device))

    test_metrics = evaluate(model, test_loader, device)
    print("\n=== Test Metrics ===")
    print(f"loss      : {test_metrics['loss']:.4f}")
    print(f"accuracy  : {test_metrics['acc']:.4f}")
    print(f"precision : {test_metrics['precision']:.4f}")
    print(f"recall    : {test_metrics['recall']:.4f}")
    print(f"f1        : {test_metrics['f1']:.4f}")
    print("confusion matrix:")
    print(test_metrics["cm"])


if __name__ == "__main__":
    main()
