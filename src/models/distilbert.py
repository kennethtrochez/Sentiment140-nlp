"""
Fine-tune BERT

Input:
- models/splits/train.csv
- models/splits/test.csv

Processing:
- Randomly shuffles training data.
- Subsamples fixed size training, validation, and test subsets.
- Tokenizes tweets using a pretrained DistilBERT tokenizer.
- Fine tunes a binary classification head on top of the distilBERT.
- Evaluates performance using accuracy and F1 score.

Output:
- models/distilbert/200k
- Consol logs with training, validation, and test metrics
"""

import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df["text"].tolist()
        self.labels = df["target"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return{
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

def main():
    # Data paths
    root = Path(__file__).resolve().parents[2]
    split_dir = root/"models"/"splits"

    train_path = split_dir/"train.csv"
    test_path = split_dir/"test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Data Subset sizes
    TEST_N = 200000
    TRAIN_N = 200000
    VAL_N = 10000

    # Subsample test set for faster evaluation and shuffle training data for randomness
    test_df = test_df.sample(n=TEST_N, random_state=40).reset_index(drop=True)
    train_df = train_df.sample(frac=1.0, random_state=40).reset_index(drop=True)

    if TRAIN_N is not None:
        train_df = train_df.iloc[:TRAIN_N].copy()

    # Split training subset into training and validation partitions
    val_df = train_df.iloc[:VAL_N].copy()
    train_df = train_df.iloc[VAL_N:].copy()

    model_name = "distilbert-base-uncased"

    out_dir = root/"models"/"distilbert"/"200k"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Max token length
    MAX_LEN = 64

    train_dataset = TweetDataset(train_df, tokenizer, max_length=MAX_LEN)
    val_dataset = TweetDataset(val_df, tokenizer, max_length=MAX_LEN)
    test_dataset = TweetDataset(test_df, tokenizer, max_length=MAX_LEN)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        seed=40
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    val_metrics = trainer.evaluate()
    print("Validation metrics:", val_metrics)
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print("Test metrics:", test_metrics)
    print("Best checkpoint:", trainer.state.best_model_checkpoint)


if __name__ == "__main__":
    main()