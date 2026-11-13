import functools
import os
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_LEMMA = None
_MODE = "rule-based fallback"
_INITIALIZED = False


def _init():
    global _LEMMA, _MODE, _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer

        # Check local repository nltk_data directory first
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(pkg_dir)
        local_nltk = os.path.join(repo_dir, "nltk_data")
        if os.path.exists(local_nltk) and local_nltk not in nltk.data.path:
            nltk.data.path.insert(0, local_nltk)

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)

        test = WordNetLemmatizer()
        test.lemmatize("patients")
        _LEMMA = test
        _MODE = "NLTK WordNetLemmatizer"
    except Exception:
        _LEMMA = None
        _MODE = "rule-based fallback"


def lemmatization_mode():
    if not _INITIALIZED:
        _init()
    return _MODE


def _fallback(w):
    if len(w) > 5 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("sses"):
        return w[:-2]
    if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        s = w[:-3]
        return s[:-1] if len(s) > 2 and s[-1] == s[-2] else s
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    return w


@functools.lru_cache(maxsize=100000)
def _lemmatize_word(word):
    if _LEMMA:
        return _LEMMA.lemmatize(word)
    return _fallback(word)


def preprocess_text(value):
    if not _INITIALIZED:
        _init()
    s = str(value).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", " ", s)
    s = s.translate(str.maketrans({c: " " for c in string.punctuation}))
    words = re.findall(r"[a-z]+", s)
    out = []
    for w in words:
        if w in ENGLISH_STOP_WORDS or len(w) < 2:
            continue
        lemma = _lemmatize_word(w)
        if lemma and lemma not in ENGLISH_STOP_WORDS:
            out.append(lemma)
    return " ".join(out)

