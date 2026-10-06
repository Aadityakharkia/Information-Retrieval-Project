"""Text preprocessing module for Information Retrieval.
Implements tokenization, Porter stemming, stopword filtering, and sentence splitting.
"""
import re
from typing import List, Dict
from collections import Counter
from nltk.stem import PorterStemmer

# Embedded standard English stopwords list (IR standard 179 words)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", 
    "yours", "yourself", "yourselves"
}

_stemmer = PorterStemmer()

def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize text into lowercase alphanumeric words.
    
    Args:
        text: Input string
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        List of cleaned token strings
    """
    if not text:
        return []
    # Alphanumeric tokens of length >= 2
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*\b", text.lower())
    # Clean trailing or leading punctuation
    tokens = [t.strip(".,;:?!'\"-") for t in tokens if len(t.strip(".,;:?!'\"-")) >= 2]
    
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
        
    return tokens

def stem_token(token: str) -> str:
    """Apply Porter Stemmer to a single token."""
    return _stemmer.stem(token)

def tokenize_and_stem(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize and apply Porter stemming to each token.
    
    Args:
        text: Input string
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        List of stemmed terms
    """
    tokens = tokenize(text, remove_stopwords=remove_stopwords)
    return [_stemmer.stem(t) for t in tokens]

def compute_tf(terms: List[str]) -> Dict[str, int]:
    """Compute raw term frequency dictionary for a sequence of terms."""
    return dict(Counter(terms))

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences cleanly, respecting medical abbreviations.
    
    Args:
        text: Text containing multiple sentences
        
    Returns:
        List of trimmed non-empty sentences
    """
    if not text:
        return []
    
    # Replace common abbreviations temporarily to avoid splitting
    abbrevs = {
        r"\bDr\.": "Dr<DOT>",
        r"\bMr\.": "Mr<DOT>",
        r"\bMrs\.": "Mrs<DOT>",
        r"\bMs\.": "Ms<DOT>",
        r"\be\.g\.": "eg<DOT>",
        r"\bi\.e\.": "ie<DOT>",
        r"\bvs\.": "vs<DOT>",
        r"\bal\.": "al<DOT>",
        r"\bFig\.": "Fig<DOT>",
        r"\bNo\.": "No<DOT>"
    }
    
    sanitized = text
    for pattern, repl in abbrevs.items():
        sanitized = re.sub(pattern, repl, sanitized, flags=re.IGNORECASE)
    
    # Split on sentence boundaries: period, question mark, or exclamation point followed by space,
    # but NOT if immediately followed by citation marker [C1]..[C99]
    raw_sentences = re.split(r"(?<=[.!?])\s+(?!\[C\d+\])", sanitized)
    
    sentences = []
    for s in raw_sentences:
        # Restore abbreviation dots
        restored = s.replace("<DOT>", ".").strip()
        if len(restored) > 0:
            # If a token is just a solitary citation marker like "[C1]", attach it to previous sentence
            if re.fullmatch(r"\[C\d+\](?:\s*\[C\d+\])*", restored) and sentences:
                sentences[-1] = f"{sentences[-1]} {restored}"
            else:
                sentences.append(restored)
            
    return sentences
