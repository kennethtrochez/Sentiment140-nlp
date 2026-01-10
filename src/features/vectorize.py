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

def main():
    root = Path(__file__).resolve().parents[2]
    processed_path = root/"data"/"processed"/"sentiment140.csv"
    df = pd.read_csv(processed_path)
    df.dropna(subset=["target", "text"])
    #print(df.head())
    
    x = df["text"]
    y= df["target"]

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

    print("X_train_tfidf shape:", x_train_tfidf.shape)
    print("X_test_tfidf shape:", x_test_tfidf.shape)

if __name__ == "__main__":
    main()

