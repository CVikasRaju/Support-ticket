\# Support Ticket Classification \& Prioritization 🎫



\## 📖 About the Project

Customer support teams receive thousands of tickets daily. Sorting them manually wastes valuable time and delays urgent issues. This project automates support operations by using Natural Language Processing (NLP) and Machine Learning (ML) to read ticket descriptions and automatically predict:

1\. \*\*Category\*\* (e.g., Billing, Technical Issue, Product Inquiry)

2\. \*\*Priority\*\* (e.g., Low, Medium, High, Critical)



This serves as a decision-support system to reduce backlog and improve customer satisfaction.



\## 🚀 Tech Stack

\* \*\*Language:\*\* Python 3.10+

\* \*\*Data Manipulation:\*\* `pandas`, `numpy`

\* \*\*NLP Processing:\*\* `nltk`, `spacy`

\* \*\*Machine Learning:\*\* `scikit-learn` (TF-IDF, Classifiers)

\* \*\*Web UI (Optional but recommended):\*\* `streamlit`



\## 📊 Dataset Options

The agent is instructed to use the \[Customer Support Ticket Dataset (Kaggle)](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) or a similar dataset containing a `Ticket Description`, `Ticket Type` (Category), and `Ticket Priority`.



\## ⚙️ Setup Instructions

1\. Clone this repository.

2\. Create a virtual environment: `python -m venv venv`

3\. Activate the environment: `source venv/bin/activate` (Mac/Linux) or `venv\\Scripts\\activate` (Windows)

4\. Install dependencies: `pip install -r requirements.txt`

5\. Run the web application: `streamlit run app.py`

