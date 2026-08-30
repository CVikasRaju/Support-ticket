# 🎫 Support Ticket Classification & Prioritization

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="sklearn"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/NLP-TF--IDF%20%7C%20Transformers-009688?style=for-the-badge" alt="NLP"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  <b>Dual-output NLP pipeline that reads raw support tickets and predicts <code>Category</code> + <code>Priority</code> in real-time.</b><br/>
  <sub>TF-IDF + Logistic Regression / Random Forest &bull; Sentence-Transformers &bull; Streamlit &bull; 80.9% on real Zenodo data (vs 79% reference)</sub>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-results">Results</a> •
  <a href="#-project-structure">Structure</a>
</p>

---

## 📖 Overview

Customer support teams receive **thousands of tickets daily**. Manual triage wastes time and delays urgent issues (`Critical` outages buried under `Low` inquiries).

This project is a **decision-support system** that automates triage:

| Input | Output 1 — Category | Output 2 — Priority |
|-------|---------------------|---------------------|
| *"System is down, no one can access HR share, urgent!"* | `Technical Issue` | 🔴 `Critical` |
| *"What is included in premium plan?"* | `Product Inquiry` | 🟢 `Low` |
| *"Charged twice for subscription"* | `Billing` | 🟠 `High` |

**Two tasks, one pipeline:** text → `clean_text()` → vectorization → two independent classifiers → Streamlit UI with confidence scores.

> Built as an internship project with 3 training tracks: **TF-IDF baseline**, **Sentence-Transformers (all-MiniLM-L6-v2)**, and **real-world Zenodo validation** beating the authors' 79% baseline.

---

## ✨ Key Features

- **🧹 Robust NLP Preprocessing** — lowercasing, regex noise removal, NLTK stopwords, WordNet lemmatization
- **🔢 Dual Feature Strategies** — `TfidfVectorizer(max_features=5000)` + `sentence-transformers/all-MiniLM-L6-v2` (384-d embeddings)
- **🤖 Task-specific Models** — `LogisticRegression (balanced)` for Category, `RandomForest (n=200)` for Priority, `LinearSVC + SelectKBest(chi2) + GridSearchCV` for real data
- **📊 Full Evaluation** — accuracy / precision / recall / F1 (weighted), `classification_report`, confusion matrices (`seaborn`)
- **🎨 Interactive UI** — Streamlit app with color-coded priority (`Critical=red`, `High=orange`, `Medium=gold`, `Low=green`), per-class probability bars
- **🔒 Privacy by Default** — `.gitignore` excludes `data/`, `models/*.pkl`, `results/`, `.env`, `secrets.toml`, `.kaggle/kaggle.json`
- **🧪 Reproducible** — `random_state=42`, stratified 80/20 split, synthetic data generator (6000 rows, label-noise simulation)

---

## 🖥️ Demo

```bash
streamlit run app.py
# → http://localhost:8501
```

1. Paste ticket description: *"I cannot log in, server error after password"*
2. Click **🔍 Classify Ticket**
3. See `Category: Technical Issue (98%)` + `Priority: High (87%)` with confidence distributions

> **Models not found?** Run `python src/train.py` first — it generates `models/tfidf_vectorizer.pkl`, `category_model.pkl`, `priority_model.pkl`.

<details>
<summary>📸 Screenshots</summary>

- `results/category_confusion_matrix.png` — 100% Category accuracy (synthetic)
- `results/priority_confusion_matrix.png` — 94.3% Priority accuracy
- `results/category_confusion_matrix_real.png` — 80.9% on Zenodo real data
- `results/emb_category_cm.png` / `emb_priority_cm.png` — embedding track

</details>

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Raw Ticket Text] --> B[clean_text()\n lower + regex + stopwords + lemmatize]
    B --> C{Feature Extraction}
    C -->|TF-IDF 5000| D[LogisticRegression\nCategory]
    C -->|TF-IDF 5000| E[RandomForest n=200\nPriority]
    C -->|MiniLM 384-d| F[Best of LogReg / LinearSVC / RF]
    D --> G[Streamlit UI\n Confidence + Color]
    E --> G
    F --> G
```

**Pipeline (per `docs/Project_specs.md`):**

1. **Preprocess** (`src/preprocess.py`) — `clean_text(text: str) -> str`
2. **Load & Split** (`src/data_loader.py`) — combines `Ticket Subject + Description → text`, drops NA, `train_test_split(stratify=Ticket Type)`
3. **Vectorize** — `TfidfVectorizer` fit on train only (no leakage) → `joblib.dump`
4. **Train** — Category=`LogisticRegression(class_weight='balanced')`, Priority=`RandomForest(class_weight='balanced')`
5. **Evaluate** (`src/evaluate.py`) — `metrics.json` + heatmaps
6. **Inference** (`app.py`) — `@st.cache_resource load_models()` + `predict_proba`

---

## 📦 Project Structure

```
Support Ticket Classification & Prioritization/
├── app.py                        # Streamlit inference UI
├── requirements.txt
├── .gitignore                    # protects data/, models/, secrets, .env
├── README.md                     # ← you are here
├── data/
│   ├── customer_support_tickets.csv   # synthetic 6000 rows (generated)
│   └── zenodo/                       # real Zenodo validation set + pretrained .h5
│       ├── X_train.csv / X_test.csv
│       └── y_train.csv / y_test.csv
├── models/                       # gitignored — regenerable
│   ├── tfidf_vectorizer.pkl
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   ├── *_real.pkl / *_emb.pkl
│   └── sentence_model/           # MiniLM safetensors (80MB)
├── results/                      # gitignored — confusion matrices + metrics
│   ├── metrics.json / metrics_real.json / metrics_emb.json
│   └── *.png
├── src/
│   ├── data_loader.py            # load_data(), preprocess_data(), split_data()
│   ├── preprocess.py             # clean_text()
│   ├── train.py                  # TF-IDF baseline (main track)
│   ├── train_embeddings.py       # Sentence-Transformers track
│   ├── train_real.py             # Zenodo + GridSearchCV + SelectKBest
│   ├── evaluate.py               # metrics + plots
│   ├── generate_synthetic_data.py# template-based generator with noise
│   ├── priority_rules.py         # keyword heuristic fallback
│   └── verify_real.py            # spot-check real model
└── docs/
    ├── Project_specs.md
    ├── TASKS.md
    └── README.md                 # original brief
```

---

## 📊 Dataset

| Source | Rows | Columns | Description |
|--------|------|---------|-------------|
| **Synthetic** (`generate_synthetic_data.py`) | 6000 | `Ticket Subject, Ticket Description, Ticket Type, Ticket Priority` | Template-based, 5 categories × 8-10 intents × 3 phrasings, urgent/calm augmentation, 6% neighbor label flip |
| **Kaggle** ([Customer Support Tickets](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)) | — | `Ticket Description, Ticket Type, Ticket Priority` | Expected schema for `data_loader.py` |
| **Zenodo Real** (`data/zenodo/`) | 2629 train / 657 test | `text, category_truth (7 classes)` | German IT tickets, used in `train_real.py` |

**Categories (synthetic):** `Billing`, `Technical Issue`, `Product Inquiry`, `Account Management`, `Cancellation`
**Categories (real):** `EOL, Fileservice, O365, Active Directory, Computer-Services, Software, Support general`
**Priorities:** `Critical`, `High`, `Medium`, `Low`

---

## 🧰 Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.10+ |
| **Data** | `pandas`, `numpy` |
| **NLP** | `nltk` (stopwords, WordNet), `spacy`, `sentence-transformers`, `TfidfVectorizer` |
| **ML** | `scikit-learn` (LogisticRegression, RandomForest, LinearSVC, GridSearchCV, SelectKBest/chi2, CalibratedClassifierCV) |
| **Viz** | `matplotlib`, `seaborn` |
| **App** | `streamlit`, `joblib` |
| **Optional** | `torch` (via sentence-transformers) |

---

## 📈 Results

### Synthetic — TF-IDF Baseline (`src/train.py` + `src/evaluate.py`)

| Model | Accuracy | Precision (w) | Recall (w) | F1 (w) | Notes |
|-------|----------|---------------|------------|--------|-------|
| **Category — LogisticRegression** | **1.000** | 1.000 | 1.000 | 1.000 | 1200 test, 5 classes |
| **Priority — RandomForest (n=200)** | **0.943** | 0.943 | 0.943 | 0.943 | Critical F1 0.90, Low 0.95, Medium 0.95 |

> Synthetic templates are separable by design — 100% Category is expected; Priority 94% reflects intentional label noise.

### Synthetic — Embeddings (`src/train_embeddings.py` — MiniLM-L6-v2)

| Model | Accuracy | F1 (w) |
|-------|----------|--------|
| **Category** (best: LogReg) | **1.000** | 1.000 |
| **Priority** (best: LogReg) | **0.953** | 0.953 | +1% over TF-IDF, Critical recall 0.89 → precision 0.99 |

### Real — Zenodo (`src/train_real.py` — TF-IDF + SelectKBest + LinearSVC GridSearch)

| Metric | Score | Reference |
|--------|-------|-----------|
| **Accuracy** | **0.810** | 0.79 (authors) ✅ |
| Precision (w) | 0.808 | — |
| Recall (w) | 0.810 | — |
| F1 (w) | 0.804 | — |

Best params: `max_features=5000-20000, ngram_range=(1,2), sublinear_tf=True, k=3000-6000, C=0.1/1.0` • Per-class F1: `EOL 1.00, Fileservice 0.94, Support general 0.82, Computer-Services 0.79, O365 0.69, Software 0.64, Active Directory 0.56`

Artifacts: `results/metrics*.json`, `*confusion_matrix*.png`, `emb_*_cm.png`

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ • `pip` • (optional) `venv`

### 1. Clone & Environment

```bash
git clone <your-repo-url>
cd "Support Ticket Classification & Prioritization"

# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install

```bash
pip install --upgrade pip
pip install -r requirements.txt

# NLTK data (auto-downloaded in preprocess.py, or manual):
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 3. Get Data

**Option A — Generate synthetic (no download):**
```bash
python src/generate_synthetic_data.py
# → data/customer_support_tickets.csv (6000 rows)
```

**Option B — Kaggle (realistic):**
1. Download [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)
2. Place as `data/customer_support_tickets.csv` with columns `Ticket Subject, Ticket Description, Ticket Type, Ticket Priority`

**Option C — Zenodo (already included in `data/zenodo/`):**
- `X_train.csv, X_test.csv, y_train.csv, y_test.csv` ready for `train_real.py`

### 4. Train

```bash
# Baseline (TF-IDF) — main track, ~10s
python src/train.py
# → models/tfidf_vectorizer.pkl, category_model.pkl, priority_model.pkl
# → results/test_data.csv

# Evaluate baseline
python src/evaluate.py
# → results/metrics.json + confusion matrices
# Expected: Category 1.00, Priority 0.94

# Optional: embeddings track (~2 min + model download 80MB)
python src/train_embeddings.py
# → models/sentence_model/, *_emb.pkl, results/metrics_emb.json

# Optional: real Zenodo + GridSearch (~1-2 min)
python src/train_real.py
# → models/*_real.pkl, results/metrics_real.json (0.81 acc)
```

### 5. Run App

```bash
streamlit run app.py
# Local: http://localhost:8501
# Network: http://<your-ip>:8501
```

Paste any ticket, click **Classify Ticket**, observe color-coded priority + confidence bars.

> **Heads up:** `streamlit_out.log` is gitignored. `models/` and `data/` are gitignored — run training scripts to regenerate.

---

## 🔧 Usage Examples

### Python API

```python
from src.preprocess import clean_text
import joblib

vec = joblib.load("models/tfidf_vectorizer.pkl")
cat_model = joblib.load("models/category_model.pkl")
pri_model = joblib.load("models/priority_model.pkl")

text = "My invoice shows duplicate charge, need refund urgently"
cleaned = clean_text(text)
X = vec.transform([cleaned])

print(cat_model.predict(X)[0])        # Billing
print(pri_model.predict(X)[0])        # High
print(cat_model.predict_proba(X).max())  # confidence
```

### Priority Heuristic (no ML)

```python
from src.priority_rules import predict_priority

predict_priority("Server is DOWN, production outage!", category="Fileservice")
# → "Critical"
```

### Batch Evaluation

```bash
python src/evaluate.py          # TF-IDF metrics
python src/verify_real.py       # spot-check 15 random Zenodo samples
```

---

## 🔒 Security & .gitignore

Sensitive / regenerable paths are **never committed**:

```gitignore
data/  results/  models/  *.pkl  *.h5  *.pickle
.env  *.env  secrets.toml  .streamlit/secrets.toml
kaggle.json  .kaggle/  credentials.json  *.key  *.pem
```

Before pushing, verify:

```bash
git status --ignored   # should show data/, models/, results/ as ignored
git check-ignore -v data/customer_support_tickets.csv
```

To share **example** env, commit only `.env.example` (template without values):

```ini
# .env.example — safe to commit
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

---

## 🗺️ Roadmap

- [ ] Add `src/features.py` abstraction (currently inline in `train.py`)
- [ ] MultiOutputClassifier joint training
- [ ] FastAPI `/predict` endpoint + Docker
- [ ] MLflow experiment tracking
- [ ] Class-imbalance study (SMOTE vs `class_weight`)
- [ ] Multilingual tickets (German Zenodo → XLM-R)

---

## 🤝 Contributing

1. Fork → branch `feat/your-feature`
2. `pip install -r requirements.txt` + run `python src/train.py && python src/evaluate.py`
3. Commit with conventional messages (`feat:`, `fix:`, `docs:`)
4. PR with metrics delta + confusion matrix

---

## 📄 License

MIT — free for personal & commercial use. See `LICENSE` (add one if missing: `gh repo view --json licenseInfo`).

---

## 🙏 Acknowledgements

- Dataset: [Kaggle — Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) + Zenodo German IT corpus (`data/zenodo/`)
- Libraries: `scikit-learn`, `nltk`, `sentence-transformers`, `streamlit`
- Reference accuracy 0.79 from Zenodo `metrics.txt` / `training.png`
- Built with ☕ for support teams who deserve fewer backlogs.

<p align="center"><sub>Run <code>streamlit run app.py</code> • Train with <code>python src/train.py</code> • Evaluate with <code>python src/evaluate.py</code></sub></p>
