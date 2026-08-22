import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)

from preprocess import clean_text

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "zenodo")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_zenodo():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"), encoding="latin-1")
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"), encoding="latin-1")
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"), encoding="latin-1")
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"), encoding="latin-1")
    y_train_map = y_train.set_index("id")["category_truth"]
    y_test_map = y_test.set_index("id")["category_truth"]
    train_text = X_train["text"].apply(clean_text)
    test_text = X_test["text"].apply(clean_text)
    y_train_lbl = X_train["id"].map(y_train_map)
    y_test_lbl = X_test["id"].map(y_test_map)
    train_mask = y_train_lbl.notna()
    test_mask = y_test_lbl.notna()
    return (train_text[train_mask], y_train_lbl[train_mask],
            test_text[test_mask], y_test_lbl[test_mask])


def main():
    X_tr, y_tr, X_te, y_te = load_zenodo()
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")
    print(f"Classes: {sorted(y_tr.unique())}")

    base = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("kbest", SelectKBest(chi2)),
        ("clf", LinearSVC(max_iter=10000)),
    ])

    grid = {
        "tfidf__max_features": [5000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__sublinear_tf": [True],
        "kbest__k": [3000, 6000],
        "clf": [LinearSVC(max_iter=10000)],
        "clf__C": [0.1, 1.0],
    }

    search = GridSearchCV(base, grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    search.fit(X_tr, y_tr)

    print("\n=== Best pipeline (CV search) ===")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    y_pred = search.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n=== TEST SET (held-out) ===")
    print(f"Test accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_te, y_pred, zero_division=0))

    print("Reference (authors' model): 0.79 accuracy")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics = {
        "reference_accuracy": 0.79,
        "accuracy": acc,
        "precision_weighted": precision_score(y_te, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_te, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_te, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(y_te, y_pred, output_dict=True, zero_division=0),
    }
    with open(os.path.join(RESULTS_DIR, "metrics_real.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Saved -> results/metrics_real.json")

    if acc >= 0.75:
        vectorizer = search.best_estimator_.named_steps["tfidf"]
        clf = search.best_estimator_.named_steps["clf"]
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer_real.pkl"))
        joblib.dump(clf, os.path.join(MODELS_DIR, "category_model_real.pkl"))
        print(f"\nSaved -> models/category_model_real.pkl (+ vectorizer_real)")

    cm = confusion_matrix(y_te, y_pred, labels=sorted(y_te.unique()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(y_te.unique()), yticklabels=sorted(y_te.unique()))
    plt.title("Category Model Confusion Matrix (Real Zenodo Data)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "category_confusion_matrix_real.png"))
    plt.close()
    print("Saved -> results/category_confusion_matrix_real.png")


if __name__ == "__main__":
    main()