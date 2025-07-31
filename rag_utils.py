import re
import unicodedata
from difflib import SequenceMatcher
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def normalize_answer(ans: str) -> str:
    ans = ans.lower().strip()
    ans = unicodedata.normalize("NFKD", ans)
    ans = re.sub(r"[₹$,€]", "", ans)
    ans = re.sub(r"(crore|cr|billion|million|lakhs|lakh|usd|inr|rs)", "", ans)
    ans = re.sub(r"[^0-9a-zA-Z.\s]", "", ans)
    ans = re.sub(r"\s+", " ", ans)
    return ans.strip()

def fuzzy_match(pred: str, truth: str) -> float:
    pred_norm = normalize_answer(pred)
    truth_norm = normalize_answer(truth)
    return SequenceMatcher(None, pred_norm, truth_norm).ratio()

def get_eval_metrics(pred, truth, threshold=0.85):
    score = fuzzy_match(pred, truth)
    y_true = [1]
    y_pred = [1 if score >= threshold else 0]

    return {
        "fuzzy": round(score, 2),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
