import re
from preprocess import clean_text

CRITICAL_KEYWORDS = [
    "crash", "crashed", "outage", "down", "urgent", "emergency", "immediately",
    "asap", "locked out", "lockout", "security", "breach", "hacked", "ransomware",
    "data loss", "server down", "system down", "production down", "not accessible",
    "cannot access", "unavailable", "network down", "hard down", "critical",
    "fire", "flood", "power outage",
]

HIGH_KEYWORDS = [
    "broken", "not working", "does not work", "error", "failed", "failure",
    "wrong", "incorrect", "cannot login", "cannot log", "can't log", "can't login",
    "access denied", "permission", "twice", "double", "overcharged", "duplicate",
    "freeze", "frozen", "slow", "lag", "out of memory", "disk full", "corrupt",
    "spam", "virus", "malware", "weather", "violation",
]

MEDIUM_KEYWORDS = [
    "request", "requested", "help", "question", "need", "update", "upgrade",
    "install", "installation", "setup", "configure", "troubleshoot", "sync",
    "forwarded", "verification", "pending", "status", "issue with", "problem",
]

LOW_KEYWORDS = [
    "information", "informative", "change", "changes", "detail", "details",
    "confirm", "confirmation", "when", "schedule", "scheduled", "plan",
    "planning", "feature", "question about", "inquiry", "enquiry", "check",
    "notification",
]

CATEGORY_BASE_PRIORITY = {
    "EOL": "Medium",
    "Fileservice": "Medium",
    "O365": "Medium",
    "Active Directory": "Medium",
    "Computer-Services": "Low",
    "Software": "Medium",
    "Support general": "Medium",
}

PRIORITY_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def _score(text: str) -> dict:
    lower = text.lower()
    scores = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    for kw in CRITICAL_KEYWORDS:
        if kw in lower:
            scores["Critical"] += 1
    for kw in HIGH_KEYWORDS:
        if kw in lower:
            scores["High"] += 1
    for kw in MEDIUM_KEYWORDS:
        if kw in lower:
            scores["Medium"] += 1
    for kw in LOW_KEYWORDS:
        if kw in lower:
            scores["Low"] += 1
    return scores


def predict_priority(text: str, category: str = "") -> str:
    """Estimate ticket priority using transparent keyword + category rules."""
    if not isinstance(text, str) or not text.strip():
        return "Medium"

    cleaned = clean_text(text)
    scores = _score(cleaned)

    base = CATEGORY_BASE_PRIORITY.get(category, "Medium")
    base_idx = PRIORITY_ORDER[base]

    total = sum(scores.values())
    if total == 0:
        return base

    dominant = max(scores, key=lambda k: scores[k])
    dom_idx = PRIORITY_ORDER[dominant]

    if dominant == "Critical":
        return "Critical"
    if dom_idx >= base_idx:
        return dominant
    if base_idx > PRIORITY_ORDER[dominant]:
        return base
    return dominant


if __name__ == "__main__":
    samples = [
        ("The file server is DOWN and no one can access the HR share, urgent!", "Fileservice"),
        ("Please schedule an install of the new software on my machine next week.", "Software"),
        ("I keep getting an error and my account is locked after password attempts.", "Active Directory"),
        ("What features are included in the premium license?", "Software"),
    ]
    for text, cat in samples:
        print(f"PRIORITY: {predict_priority(text, cat):<10} CATEGORY: {cat:<18} TEXT: {text}")