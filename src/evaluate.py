import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocess import clean_text

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def plot_confusion_matrix(cm, labels, title, filename):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def compute_metrics(y_true, y_pred):
    labels = sorted(set(y_true))
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    category_model = joblib.load(os.path.join(MODELS_DIR, "category_model.pkl"))
    priority_model = joblib.load(os.path.join(MODELS_DIR, "priority_model.pkl"))

    test_df = pd.read_csv(os.path.join(RESULTS_DIR, "test_data.csv"))

    X_test = test_df['text'].apply(clean_text)
    y_cat_true = test_df['Ticket Type']
    y_pri_true = test_df['Ticket Priority']

    X_test_vec = vectorizer.transform(X_test)

    y_cat_pred = category_model.predict(X_test_vec)
    y_pri_pred = priority_model.predict(X_test_vec)

    cat_metrics = compute_metrics(y_cat_true, y_cat_pred)
    pri_metrics = compute_metrics(y_pri_true, y_pri_pred)

    cm_cat = confusion_matrix(y_cat_true, y_cat_pred, labels=sorted(set(y_cat_true)))
    cm_pri = confusion_matrix(y_pri_true, y_pri_pred, labels=sorted(set(y_pri_true)))

    plot_confusion_matrix(cm_cat, sorted(set(y_cat_true)), 'Category Model Confusion Matrix', 'category_confusion_matrix.png')
    plot_confusion_matrix(cm_pri, sorted(set(y_pri_true)), 'Priority Model Confusion Matrix', 'priority_confusion_matrix.png')

    metrics = {
        'category_model': cat_metrics,
        'priority_model': pri_metrics,
    }

    with open(os.path.join(RESULTS_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("=== CATEGORY MODEL ===")
    print(f"Accuracy: {cat_metrics['accuracy']:.4f}")
    print(f"Precision: {cat_metrics['precision']:.4f}")
    print(f"Recall: {cat_metrics['recall']:.4f}")
    print(f"F1-Score: {cat_metrics['f1']:.4f}")

    print("\n=== PRIORITY MODEL ===")
    print(f"Accuracy: {pri_metrics['accuracy']:.4f}")
    print(f"Precision: {pri_metrics['precision']:.4f}")
    print(f"Recall: {pri_metrics['recall']:.4f}")
    print(f"F1-Score: {pri_metrics['f1']:.4f}")

    print(f"\nMetrics saved -> {os.path.join(RESULTS_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()