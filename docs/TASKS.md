\# Implementation Checklist for AI Agent



\## Phase 1: Setup \& Data Ingestion

\- [x] \*\*Task 1.1:\*\* Create `requirements.txt` containing: `pandas`, `numpy`, `scikit-learn`, `nltk`, `spacy`, `joblib`, `streamlit`, `matplotlib`, `seaborn`.

\- [x] \*\*Task 1.2:\*\* Write a script `src/data\_loader.py` to load the `customer\_support\_tickets.csv` dataset, drop rows with missing descriptions or targets, and combine `Ticket Subject` and `Ticket Description` into a single `text` column.

\- [x] \*\*Task 1.3:\*\* Split the dataset into training (80%) and testing (20%) sets.



\## Phase 2: NLP Preprocessing

\- [x] \*\*Task 2.1:\*\* Create `src/preprocess.py`. Write a function `clean\_text(text)` that performs lowercasing, removes punctuation, removes NLTK stopwords, and applies lemmatization.

\- [x] \*\*Task 2.2:\*\* Apply `clean\_text` to the training and testing datasets.



\## Phase 3: Feature Engineering \& Modeling

\- [x] \*\*Task 3.1:\*\* Create `src/train.py`. Initialize a `TfidfVectorizer` (max\_features=5000), fit it on the training text, and transform both train and test sets. Save the vectorizer using `joblib`.

\- [x] \*\*Task 3.2:\*\* Train a `LogisticRegression` or `LinearSVC` model to predict the `Ticket Type` (Category). Save the model as `category\_model.pkl`.

\- [x] \*\*Task 3.3:\*\* Train a `RandomForestClassifier` (handling class weight) to predict `Ticket Priority`. Save the model as `priority\_model.pkl`.



\## Phase 4: Evaluation

\- [x] \*\*Task 4.1:\*\* Create `src/evaluate.py`. Load the test sets and the saved models.

\- [x] \*\*Task 4.2:\*\* Generate a `classification\_report` for both the Category and Priority models.

\- [x] \*\*Task 4.3:\*\* Plot a confusion matrix for both models using `seaborn` and save them as `.png` files in a `results/` folder. Write the raw metrics to `results/metrics.json`.



\## Phase 5: Deployment Simulation

\- [x] \*\*Task 5.1:\*\* Create `app.py` using Streamlit.

\- [x] \*\*Task 5.2:\*\* Build a UI with a text area for users to paste a customer complaint.

\- [x] \*\*Task 5.3:\*\* On button click, pass the text through the `clean\_text` function, vectorize it, and display the predictions for both Category and Priority in a clean, color-coded format (e.g., Red for Critical priority).

