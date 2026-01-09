"""
Content:
It contains the following 6 fields:

1. target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
2. ids: The id of the tweet ( 2087)
3. date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
4. flag: The query (lyx). If there is no query, then this value is NO_QUERY.
5. user: the user that tweeted (robotickilldozr)
6. text: the text of the tweet (Lyx is cool)
"""


import pandas as pd
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[2]
    csv_path = root/"data"/"raw"/"training.1600000.processed.noemoticon.csv"

    cols = ["target", "ids", "date", "flag", "user", "text"]

    # latin-1 encoding over UTF-8 because Text is from 2009
    df = pd.read_csv(
        csv_path,
        header=None,
        names=cols,
        encoding="latin-1"
    )


    # Target & Text
    df = df[["target", "text"]]
    # print(df["target"].unique())
    df["target"] = df["target"].map({0: 0, 4: 1})
    df = df.dropna(subset=["target"])

    df.to_csv("data/processed/sentiment140.csv", index=False)


if __name__ == "__main__":
    main()