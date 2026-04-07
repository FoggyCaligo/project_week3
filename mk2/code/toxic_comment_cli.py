import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEFAULT_MODEL = "beomi/KcELECTRA-base"
DEFAULT_MAX_LENGTH = 128


@dataclass
class LabelConfig:
    id2label: Dict[int, str]
    label2id: Dict[str, int]


def get_label_config() -> LabelConfig:
    id2label = {0: "normal", 1: "toxic"}
    label2id = {v: k for k, v in id2label.items()}
    return LabelConfig(id2label=id2label, label2id=label2id)


def prepare_kold() -> DatasetDict:
    ds = load_dataset("nayohan/KOLD")
    base = ds["train"]

    def map_fn(example):
        return {
            "text": example["comment"],
            "label": int(bool(example["OFF"])),
            "title": example.get("title", ""),
            "source": example.get("source", ""),
        }

    base = base.map(map_fn)
    keep_cols = [c for c in ["text", "label", "title", "source"] if c in base.column_names]
    base = base.remove_columns([c for c in base.column_names if c not in keep_cols])

    # train_test_split(..., stratify_by_column=...) requires ClassLabel, not plain Value.
    base = base.class_encode_column("label")

    split_1 = base.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")

    return DatasetDict(
        {
            "train": split_1["train"],
            "validation": split_2["train"],
            "test": split_2["test"],
        }
    )


NOT_HATE_LABEL = 8


def prepare_kmhas() -> DatasetDict:
    ds = load_dataset("jeanlee/kmhas_korean_hate_speech")

    def map_fn(example):
        labels = example["label"]
        is_toxic = 0 if (len(labels) == 1 and labels[0] == NOT_HATE_LABEL) else 1
        return {"text": example["text"], "label": is_toxic}

    out = DatasetDict()
    for split in ["train", "validation", "test"]:
        mapped = ds[split].map(map_fn)
        mapped = mapped.remove_columns([c for c in mapped.column_names if c not in ["text", "label"]])
        out[split] = mapped.class_encode_column("label")
    return out


def load_binary_dataset(dataset_name: str) -> DatasetDict:
    dataset_name = dataset_name.lower()
    if dataset_name == "kold":
        return prepare_kold()
    if dataset_name == "kmhas":
        return prepare_kmhas()
    raise ValueError("dataset must be one of: kold, kmhas")


def tokenize_dataset(dataset: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset.map(tok, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train(args):
    set_seed(args.seed)
    labels = get_label_config()
    dataset = load_binary_dataset(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = tokenize_dataset(dataset, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=labels.id2label,
        label2id=labels.label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_metrics = trainer.evaluate(tokenized["validation"])
    test_metrics = trainer.evaluate(tokenized["test"])

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    result = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "validation": val_metrics,
        "test": test_metrics,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== Validation Metrics ===")
    print(json.dumps(val_metrics, ensure_ascii=False, indent=2))
    print("\n=== Test Metrics ===")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved model to: {args.output_dir}")
    print(f"Saved metrics to: {os.path.join(args.output_dir, 'metrics.json')}")


@torch.no_grad()
def predict(args):
    labels = get_label_config()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def run_one(text: str):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        print("-" * 60)
        print(f"입력: {text}")
        print(f"예측: {labels.id2label.get(pred_id, str(pred_id))}")
        print(f"normal 확률: {probs[0]:.4f}")
        print(f"toxic  확률: {probs[1]:.4f}")

    if args.text:
        run_one(args.text)
        return

    print("문장을 입력하세요. 종료하려면 엔터만 입력하세요.")
    while True:
        text = input("> ").strip()
        if not text:
            break
        run_one(text)


def build_parser():
    parser = argparse.ArgumentParser(description="Korean toxic comment detector (CLI baseline)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a binary toxic comment classifier")
    p_train.add_argument("--dataset", choices=["kold", "kmhas"], default="kold")
    p_train.add_argument("--model_name", default=DEFAULT_MODEL)
    p_train.add_argument("--output_dir", default="./outputs/toxic_kold")
    p_train.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    p_train.add_argument("--epochs", type=float, default=3)
    p_train.add_argument("--learning_rate", type=float, default=2e-5)
    p_train.add_argument("--train_batch_size", type=int, default=16)
    p_train.add_argument("--eval_batch_size", type=int, default=32)
    p_train.add_argument("--logging_steps", type=int, default=100)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=train)

    p_pred = sub.add_parser("predict", help="Predict toxicity from a saved model")
    p_pred.add_argument("--model_dir", required=True)
    p_pred.add_argument("--text", default=None)
    p_pred.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    p_pred.set_defaults(func=predict)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
