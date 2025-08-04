# rag_utils.py
import re
import unicodedata
from difflib import SequenceMatcher

def normalize_answer(ans):
    ans = ans.lower().strip()
    ans = unicodedata.normalize("NFKD", ans)
    ans = re.sub(r'[\$,\u20B9,\u20AC]', '', ans)  # remove currency
    ans = re.sub(r'[\n\r]', ' ', ans)
    ans = re.sub(r'[^\w\d\.\s]', '', ans)  # keep text and numbers
    ans = re.sub(r'\s+', ' ', ans)
    return ans

def fuzzy_match(a, b):
    a_norm = normalize_answer(a)
    b_norm = normalize_answer(b)
    return SequenceMatcher(None, a_norm, b_norm).ratio()