\# Project Specification: ML Architecture



\## 1. Objective

Build a dual-output Machine Learning pipeline that takes raw customer support text and returns a predicted Category and Priority. 



\## 2. Dataset Schema (Expected)

The pipeline should expect a CSV with at least these conceptual columns (adapt based on exact Kaggle dataset used):

\* `Ticket Subject` (String)

\* `Ticket Description` (String) - \*\*Primary Input Feature\*\*

\* `Ticket Type` / Category (String) - \*\*Target 1\*\*

\* `Ticket Priority` (String) - \*\*Target 2\*\*



\## 3. Machine Learning Pipeline Architecture

\### A. Text Preprocessing (`src/preprocess.py`)

\* \*\*Lowercasing:\*\* Convert all text to lowercase.

\* \*\*Noise Removal:\*\* Remove punctuation, special characters, and numbers.

\* \*\*Stopwords:\*\* Remove common English stopwords using NLTK or spaCy.

\* \*\*Lemmatization:\*\* Convert words to their base form.



\### B. Feature Extraction (`src/features.py`)

\* \*\*Vectorization:\*\* Use `TfidfVectorizer` from `scikit-learn`.

\* Ensure the vectorizer is fitted on the training set only to prevent data leakage, and saved as `tfidf\_vectorizer.pkl`.



\### C. Modeling (`src/train.py`)

We are treating this as two separate classification tasks (or a MultiOutputClassifier):

1\. \*\*Category Model:\*\* Multinomial Naive Bayes or LinearSVC.

2\. \*\*Priority Model:\*\* Random Forest or Logistic Regression (handles class imbalances better).

\* Both models must be serialized using `joblib` or `pickle`.



\### D. Evaluation (`src/evaluate.py`)

\* Calculate Accuracy, Precision, Recall, and F1-Score for both models.

\* Generate a Confusion Matrix for both predictions.

\* Save metrics to a `metrics.json` file.



\### E. Inference \& UI (`app.py`)

\* A Streamlit interface where a user can type a sample ticket.

\* The app loads the pickled vectorizer and models, transforms the text, and displays the predicted Category and Priority.

