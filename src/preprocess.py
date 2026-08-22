import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

REQUIRED_RESOURCES = ['stopwords', 'wordnet', 'omw-1.4']


def _ensure_resources():
    def available():
        try:
            stopwords.words('english')
            WordNetLemmatizer().lemmatize('running')
            return True
        except LookupError:
            return False

    if available():
        return

    for resource in REQUIRED_RESOURCES:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


_ensure_resources()

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Clean and normalize text data.

    Performs lowercasing, removes punctuation/special characters/numbers,
    removes NLTK stopwords, and applies lemmatization.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS]
    return ' '.join(tokens)


if __name__ == "__main__":
    sample = "Hello, my internet is not working after the update 2.0! Please fix it immediately."
    print(f"Input:  {sample}")
    print(f"Output: {clean_text(sample)}")