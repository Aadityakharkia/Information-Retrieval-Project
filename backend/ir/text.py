r"""
backend/ir/text.py
Text Preprocessing Module for HealthNest.
Implements:
1. NFKC Unicode Normalization
2. Lowercasing
3. Tokenization using Unicode regex [\w\u0900-\u097F]+
   NOTE ON REGEX: Plain \w in Python regex splits Devanagari Hindi characters
   at vowel signs (matras, range \u0900-\u097F). Including \u0900-\u097F keeps
   Devanagari tokens intact!
4. Stop-words removal (English + Hinglish lists)
5. Optional Porter Stemming (nltk.stem.porter.PorterStemmer)
6. Script & Language Identification (English vs Devanagari vs Hinglish)
"""

import re
import unicodedata
from typing import List, Tuple
from nltk.stem.porter import PorterStemmer
import config

_porter = PorterStemmer()

# Concise English Stopwords List (standard IR stop list)
ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

# Concise Hinglish Stopwords List
HINGLISH_STOP_WORDS = {
    "me", "mein", "mai", "ka", "ki", "ke", "ko", "se", "par", "pe", "hai", "hain",
    "ho", "hu", "hoon", "tha", "thi", "the", "bhi", "toh", "to", "aur", "ya",
    "kya", "kyun", "kaise", "kahan", "kab", "kaun", "kon", "ye", "yeh", "woh",
    "voh", "mera", "meri", "mere", "aap", "tum", "hum", "sab", "kuch"
}

ALL_STOP_WORDS = ENGLISH_STOP_WORDS.union(HINGLISH_STOP_WORDS)

# Regex matching alphanumeric words plus Devanagari script range \u0900-\u097F
TOKEN_REGEX = re.compile(r'[\w\u0900-\u097F]+', re.UNICODE)


def normalize_text(text: str) -> str:
    """Applies Unicode NFKC normalization and lowercasing."""
    if not text:
        return ""
    text_nfkc = unicodedata.normalize("NFKC", text)
    return text_nfkc.lower()


def tokenize(
    text: str,
    use_stopwords: bool = config.USE_STOP_WORDS,
    use_stemmer: bool = config.USE_PORTER_STEMMER
) -> List[str]:
    """
    Normalizes, tokenizes, removes stop-words, and optionally stems tokens.
    """
    norm_text = normalize_text(text)
    raw_tokens = TOKEN_REGEX.findall(norm_text)
    
    tokens = []
    for token in raw_tokens:
        if use_stopwords and token in ALL_STOP_WORDS:
            continue
        if use_stemmer and token.isascii():  # Only stem ASCII/English words
            token = _porter.stem(token)
        tokens.append(token)

    return tokens


def detect_language(text: str) -> str:
    """
    Detects query language: 'hindi' (Devanagari script), 'hinglish' (Roman Hindi words), or 'english'.
    """
    if not text:
        return "english"

    devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = max(1, len(text.strip()))
    
    if devanagari_chars / total_chars > 0.1:
        return "hindi"

    # Tokenize without stop-words filter to inspect raw words
    raw_tokens = TOKEN_REGEX.findall(normalize_text(text))
    hinglish_hits = sum(1 for t in raw_tokens if t in HINGLISH_STOP_WORDS)
    
    if hinglish_hits >= 1 or (len(raw_tokens) > 0 and hinglish_hits / len(raw_tokens) >= 0.2):
        return "hinglish"

    return "english"
