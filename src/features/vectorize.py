"""
Vectorizing text using TF-IDF

- Loading processed data
- Splitting into train/test sets at an 80/20 split
- Fit a TF-IDF vectorizer on training text only
- Transform both splits into numeric features

"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy.sparse import save_npz

def main():
    root = Path(__file__).resolve().parents[2]
    processed_path = root/"data"/"processed"/"sentiment140.csv"
    out_dir = root/"models"/"tf-idf"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(processed_path)
    df = df.dropna(subset=["target", "text"])
    #print(df.head())

    x = df["text"]
    y = df["target"]

    # Train & Test split 80/20
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=40,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),
        min_df=2
    )

    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    vectorizer_path = out_dir/"tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)

    x_train_path = out_dir/"x_train_tfidf.npz"
    x_test_path = out_dir/"x_test_tfidf.npz"

    save_npz(x_train_path, x_train_tfidf)
    save_npz(x_test_path, x_test_tfidf)


    y_train_path = out_dir/"y_train.csv"
    y_test_path = out_dir/"y_test.csv"

    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)


if __name__ == "__main__":
    main()

