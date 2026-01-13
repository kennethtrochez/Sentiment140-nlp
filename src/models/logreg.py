import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

def main():
    root = Path(__file__).resolve().parents[2]
    tfidf_dir = root/"models"/"tf-idf"

    x_train = load_npz(tfidf_dir/"x_train_tfidf.npz")
    x_test = load_npz(tfidf_dir/"x_test_tfidf.npz")

    y_train = pd.read_csv(tfidf_dir/"y_train.csv")["target"]
    y_test = pd.read_csv(tfidf_dir/"y_test.csv")["target"]

    lr_model = LogisticRegression(
        max_iter=1000
    )

    lr_model.fit(x_train, y_train)

    y_pred = lr_model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"f1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test,y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    logreg_dir = root/"models"/"logreg"
    logreg_dir.mkdir(parents=True, exist_ok=True)

    model_path = logreg_dir/"logreg_tfidf.joblib"
    joblib.dump(lr_model, model_path)

if __name__ == "__main__":
    main()