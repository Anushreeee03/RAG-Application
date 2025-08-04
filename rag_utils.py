# rag_utils.py

import re
from difflib import SequenceMatcher

def normalize_answer(text: str) -> str:
    """
    Normalize an answer string by:
    - Lowercasing
    - Removing currency symbols, commas, units
    - Converting percentages or other symbols to plain form
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower().strip()

    # Remove currency symbols and formatting
    text = re.sub(r"[₹$€£,]", "", text)

    # Remove units like 'cr', 'crore', 'million', 'percent', 'cr.', etc.
    text = re.sub(r"\b(cr|crore|million|billion|percent|cr\.|%)\b", "", text)

    # Remove extra punctuation and whitespace
    text = re.sub(r"[^\d.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def fuzzy_match(predicted: str, expected: str) -> float:
    """
    Return a similarity score (0-1) between the normalized prediction and ground truth.
    """
    pred = normalize_answer(predicted)
    exp = normalize_answer(expected)

    return SequenceMatcher(None, pred, exp).ratio()


def get_eval_metrics(predicted: str, expected: str, threshold: float = 0.85) -> dict:
    """
    Returns accuracy, precision, recall, f1 based on fuzzy match between predicted and expected.
    Used for inline evaluation inside Streamlit.
    """
    score = fuzzy_match(predicted, expected)
    y_true = 1
    y_pred = 1 if score >= threshold else 0

    accuracy = int(y_pred == y_true)
    precision = y_pred  # since y_true is always 1, precision = TP / (TP+FP) = 1 or 0
    recall = y_pred     # recall = TP / (TP+FN)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "fuzzy": round(score, 2),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": round(f1, 2)
    }
