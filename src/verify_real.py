import os
import joblib
import pandas as pd
from preprocess import clean_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data", "zenodo")


def main():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer_real.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "category_model_real.pkl"))

    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"), encoding="latin-1")
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"), encoding="latin-1")
    y_map = y_test.set_index("id")["category_truth"]
    X_test["label"] = X_test["id"].map(y_map)
    X_test = X_test.dropna()

    import random
    random.seed(1)
    sample = X_test.sample(15, random_state=1)
    correct = 0
    print(f"{'PREDICTION':<20}{'TRUTH':<20} OK  TEXT")
    for _, row in sample.iterrows():
        pred = model.predict(vectorizer.transform([clean_text(row["text"])]))[0]
        truth = row["label"]
        ok = pred == truth
        correct += ok
        print(f"{pred:<20}{truth:<20} {'YES' if ok else 'NO '}  {row['text'][:70]}")

    print(f"\nSample accuracy: {correct}/{len(sample)} = {correct / len(sample):.0%}")


if __name__ == "__main__":
    main()