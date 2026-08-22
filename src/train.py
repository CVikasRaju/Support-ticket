import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data_loader import load_data, preprocess_data, split_data
from preprocess import clean_text

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    ensure_dirs()

    df = load_data()
    df = preprocess_data(df)

    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = split_data(df)

    X_train_clean = X_train.apply(clean_text)
    X_test_clean = X_test.apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train_clean)
    X_test_vec = vectorizer.transform(X_test_clean)

    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    print("Saved TfidfVectorizer -> models/tfidf_vectorizer.pkl")

    category_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    category_model.fit(X_train_vec, y_cat_train)
    joblib.dump(category_model, os.path.join(MODELS_DIR, "category_model.pkl"))
    print("Saved category model -> models/category_model.pkl")

    priority_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    priority_model.fit(X_train_vec, y_pri_train)
    joblib.dump(priority_model, os.path.join(MODELS_DIR, "priority_model.pkl"))
    print("Saved priority model -> models/priority_model.pkl")

    pd.DataFrame({
        'text': X_test_clean,
        'Ticket Type': y_cat_test,
        'Ticket Priority': y_pri_test
    }).to_csv(os.path.join(RESULTS_DIR, "test_data.csv"), index=False)
    print("Saved test data -> results/test_data.csv")


if __name__ == "__main__":
    main()