import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)

from preprocess import clean_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(ROOT, "data", "customer_support_tickets.csv")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    df = pd.read_csv(DATA_CSV)
    df["text"] = (df["Ticket Subject"] + " " + df["Ticket Description"]).apply(clean_text)

    df_tr, df_te = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Ticket Type"])

    X_train, X_test = df_tr["text"], df_te["text"]
    y_cat_train, y_cat_test = df_tr["Ticket Type"], df_te["Ticket Type"]
    y_pri_train, y_pri_test = df_tr["Ticket Priority"], df_te["Ticket Priority"]

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(EMB_MODEL_NAME)
    print("Encoding train...")
    X_train_emb = encoder.encode(X_train.tolist(), show_progress_bar=True, batch_size=128)
    print("Encoding test...")
    X_test_emb = encoder.encode(X_test.tolist(), show_progress_bar=True, batch_size=128)

    candidates = {
        "LogReg": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "SVC-cal": CalibratedClassifierCV(LinearSVC(max_iter=10000, class_weight="balanced"), cv=5),
        "RF": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    }

    best_cat, best_pri = None, None
    cat_acc, pri_acc = 0.0, 0.0
    for name, model in candidates.items():
        for target, (y_tr, y_te) in {"cat": (y_cat_train, y_cat_test), "pri": (y_pri_train, y_pri_test)}.items():
            fitted = clone(model)
            fitted.fit(X_train_emb, y_tr)
            acc = accuracy_score(y_te, fitted.predict(X_test_emb))
            print(f"[{target.upper()}] {name:8s} test acc = {acc:.4f}")
            if target == "cat" and acc > cat_acc:
                cat_acc, best_cat = acc, fitted
            if target == "pri" and acc > pri_acc:
                pri_acc, best_pri = acc, fitted

    os.makedirs(MODELS_DIR, exist_ok=True)
    encoder.save(os.path.join(MODELS_DIR, "sentence_model"))
    joblib.dump(best_cat, os.path.join(MODELS_DIR, "category_model_emb.pkl"))
    joblib.dump(best_pri, os.path.join(MODELS_DIR, "priority_model_emb.pkl"))
    print("\nSaved models/ -> sentence_model/, category_model_emb.pkl, priority_model_emb.pkl")

    y_cat_pred = best_cat.predict(X_test_emb)
    y_pri_pred = best_pri.predict(X_test_emb)

    for target, y_true, y_pred, acc in [("CATEGORY", y_cat_test, y_cat_pred, cat_acc),
                                         ("PRIORITY", y_pri_test, y_pri_pred, pri_acc)]:
        print(f"\n=== {target} (embedding) ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=sorted(y_true.unique()), yticklabels=sorted(y_true.unique()))
        plt.title(f"{target} Confusion Matrix (Embeddings)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"emb_{target.lower()}_cm.png"))
        plt.close()

    metrics = {
        "category": {"accuracy": cat_acc,
                     "f1_weighted": f1_score(y_cat_test, y_cat_pred, average="weighted", zero_division=0),
                     "report": classification_report(y_cat_test, y_cat_pred, zero_division=0, output_dict=True)},
        "priority": {"accuracy": pri_acc,
                     "f1_weighted": f1_score(y_pri_test, y_pri_pred, average="weighted", zero_division=0),
                     "report": classification_report(y_pri_test, y_pri_pred, zero_division=0, output_dict=True)},
        "model": EMB_MODEL_NAME,
    }
    with open(os.path.join(RESULTS_DIR, "metrics_emb.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved -> results/metrics_emb.json")


if __name__ == "__main__":
    main()