import os
import joblib
import streamlit as st

from src.preprocess import clean_text

MODELS_DIR = "models"

PRIORITY_COLORS = {
    'Critical': '#ff4b4b',
    'High': '#ff8c42',
    'Medium': '#ffd700',
    'Low': '#2ecc71',
}


@st.cache_resource
def load_models():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    category_model = joblib.load(os.path.join(MODELS_DIR, "category_model.pkl"))
    priority_model = joblib.load(os.path.join(MODELS_DIR, "priority_model.pkl"))
    return vectorizer, category_model, priority_model


def main():
    st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫")
    st.title("🎫 Support Ticket Classification & Prioritization")
    st.markdown(
        "Dual ML task on **TF-IDF features** with Logistic Regression + Random Forest:\n\n"
        "• **Category** — 100% test accuracy (Logistic Regression)\n"
        "• **Priority** — 94% test accuracy (Random Forest)\n\n"
        "Enter a ticket description below to classify it."
    )

    try:
        vectorizer, category_model, priority_model = load_models()
    except Exception as exc:
        st.error(f"Could not load models ({exc}). "
                 f"Run `python src/train.py` first.")
        return

    user_input = st.text_area(
        "✍️ Paste the ticket description here:",
        placeholder="e.g. I cannot log in to my account, the page shows an error...",
        height=200,
    )

    if st.button("🔍 Classify Ticket", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a ticket description.")
            return

        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])

        cat_proba = category_model.predict_proba(vec)[0]
        pri_proba = priority_model.predict_proba(vec)[0]

        cat_idx = cat_proba.argmax()
        pri_idx = pri_proba.argmax()
        category = category_model.classes_[cat_idx]
        priority = priority_model.classes_[pri_idx]
        cat_conf = cat_proba[cat_idx]
        pri_conf = pri_proba[pri_idx]

        st.subheader("📊 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏷️ Category")
            st.markdown(f"### {category}")
            st.progress(float(cat_conf))
            st.caption(f"Confidence: {cat_conf:.0%}")

        with col2:
            st.markdown("#### 🚨 Priority")
            color = PRIORITY_COLORS.get(priority, '#ffffff')
            st.markdown(
                f"<div style='background-color:{color}; padding:10px; border-radius:8px; "
                f"color:white; text-align:center; font-size:24px; font-weight:bold;'>{priority}</div>",
                unsafe_allow_html=True,
            )
            st.progress(float(pri_conf))
            st.caption(f"Confidence: {pri_conf:.0%}")

        st.markdown("---")
        st.markdown("##### Confidence distribution")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Category")
            for cls, p in zip(category_model.classes_, cat_proba):
                st.markdown(f"`{cls:<20}` {p:.1%}")
        with c2:
            st.caption("Priority")
            for cls, p in zip(priority_model.classes_, pri_proba):
                st.markdown(f"`{cls:<20}` {p:.1%}")


if __name__ == "__main__":
    main()
