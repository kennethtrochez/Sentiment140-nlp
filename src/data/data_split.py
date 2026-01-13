import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[2]
    proccessed_path = root/"data"/"processed"/"sentiment140.csv"
    df = pd.read_csv(proccessed_path)
    df = df.dropna(subset=["target", "text"])
    out_dir = root/"models"/"splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    x = df["text"]
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=0.2,
        random_state=40,
        stratify=y
    )

    train_df = pd.DataFrame({"text": x_train, "target": y_train})
    test_df = pd.DataFrame({"text": x_test, "target": y_test})

    train_path = out_dir/"train.csv"
    test_path = out_dir/"test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

if __name__ == "__main__":
    main()