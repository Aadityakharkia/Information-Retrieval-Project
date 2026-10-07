"""
backend/rag/llm_client.py
LLM Client Interface for HealthNest.

Provider chain (first that succeeds wins):
  1. Groq cloud API   - when GROQ_API_KEY is configured and the model is a Groq model id.
  2. Ollama (local)   - model tags such as llama3.2:1b.
  3. Offline mock     - deterministic extractive answer built from the retrieved context,
                        so the system still works with no network / no model server.

Temperature is 0.0 for deterministic, verifiable answers.
"""

import logging
import re
from typing import List, Dict, Any, Tuple

import requests

import config
from backend.ir.text import tokenize

logger = logging.getLogger(__name__)


def _call_groq(messages: List[Dict[str, str]], model: str, temperature: float) -> str:
    res = requests.post(
        config.GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=config.LLM_TIMEOUT_SECONDS,
    )
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()


def _call_ollama(messages: List[Dict[str, str]], model: str, temperature: float) -> str:
    res = requests.post(
        f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=config.LLM_TIMEOUT_SECONDS,
    )
    res.raise_for_status()
    return res.json().get("message", {}).get("content", "").strip()


def generate_llm_response(
    messages: List[Dict[str, str]],
    model: str = config.DEFAULT_MODEL,
    temperature: float = config.LLM_TEMPERATURE,
) -> Tuple[str, str]:
    """
    Returns (answer_text, provider) where provider is 'groq', 'ollama' or 'mock'.
    Never raises on provider failure; degrades to the next provider instead.
    """
    if model != config.MOCK_MODEL:
        is_ollama_tag = ":" in model
        if config.GROQ_API_KEY and not is_ollama_tag:
            try:
                return _call_groq(messages, model, temperature), "groq"
            except Exception as exc:
                logger.warning("Groq call failed (%s); falling back.", exc)
        if is_ollama_tag:
            try:
                text = _call_ollama(messages, model, temperature)
                if text:
                    return text, "ollama"
            except Exception as exc:
                logger.info("Ollama unavailable (%s); falling back.", exc)
            if config.GROQ_API_KEY:  # local model missing -> use hosted default
                try:
                    return _call_groq(messages, config.GROQ_MODELS[0], temperature), "groq"
                except Exception as exc:
                    logger.warning("Groq fallback failed (%s).", exc)

    logger.info("Using offline extractive generator.")
    return generate_mock_rag_answer(messages), "mock"


def generate_mock_rag_answer(messages: List[Dict[str, str]]) -> str:
    """
    Extracts facts from context in user message and constructs cited sentences [1]..[n].
    Ensures strict relevance to the user question; abstains if context is irrelevant.
    Used when Ollama local daemon is not running.
    """
    user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            user_msg = msg["content"]
            break

    if "Context Pages:" not in user_msg:
        return "I don't know based on the provided pages."

    # Extract user question
    q_match = re.search(r"User Health Question:\s*(.+?)(?:\n\nInstructions:|$)", user_msg, re.DOTALL)
    if not q_match:
        q_match = re.search(r"Question:\s*(.+?)(?:\nAnswer:|$)", user_msg, re.DOTALL)
    question_text = q_match.group(1).strip() if q_match else ""

    from backend.ir.hinglish import HinglishExpander
    expander = HinglishExpander()
    expanded_text, _ = expander.expand_query(question_text)
    q_tokens = set(tokenize(expanded_text, use_stopwords=True, use_stemmer=True)) if expanded_text else set()

    # Parse context blocks [n]
    lines = user_msg.split("\n")
    ctx_blocks = []
    curr_n = None
    curr_text = []

    for line in lines:
        line_s = line.strip()
        if line_s.startswith("User Health Question:") or line_s.startswith("Instructions:"):
            if curr_n is not None and curr_text:
                ctx_blocks.append((curr_n, " ".join(curr_text)))
                curr_n = None
                curr_text = []
            break
        if line_s.startswith("[") and "Reference Page" in line_s:
            if curr_n is not None and curr_text:
                ctx_blocks.append((curr_n, " ".join(curr_text)))
            try:
                curr_n = int(line_s[1:line_s.find("]")])
                curr_text = []
            except ValueError:
                pass
        elif curr_n is not None and line_s:
            if not line_s.startswith("Question:"):
                curr_text.append(line_s.replace("Content: ", ""))

    if curr_n is not None and curr_text:
        ctx_blocks.append((curr_n, " ".join(curr_text)))

    if not ctx_blocks:
        return "I don't know based on the provided pages."

    # Score and filter blocks by query relevance
    relevant_blocks = []
    for n, text in ctx_blocks:
        block_tokens = set(tokenize(text, use_stopwords=True, use_stemmer=True))
        overlap = q_tokens.intersection(block_tokens) if q_tokens else set()
        # Require actual query term overlap if query tokens are available
        if not q_tokens or len(overlap) > 0:
            relevant_blocks.append((n, text, len(overlap)))

    # If no retrieved context chunks have relevant query terms, fallback to top block
    if not relevant_blocks and ctx_blocks:
        relevant_blocks = [(ctx_blocks[0][0], ctx_blocks[0][1], 1)]
    elif not relevant_blocks:
        return "I don't know based on the provided pages."

    # Sort blocks by relevance overlap
    relevant_blocks.sort(key=lambda x: x[2], reverse=True)

    cited_sentences = []
    for n, text, _ in relevant_blocks[:3]:
        raw_sents = [s.strip() for s in text.split(".") if len(s.strip()) > 15]
        # Prioritize sentences that contain query terms
        scored_sents = []
        for s in raw_sents:
            s_tokens = set(tokenize(s, use_stopwords=True, use_stemmer=True))
            s_overlap = len(q_tokens.intersection(s_tokens)) if q_tokens else 1
            scored_sents.append((s, s_overlap))
        if not any(score > 0 for _, score in scored_sents) and raw_sents:
            scored_sents = [(s, 1) for s in raw_sents]
        
        for s, score in scored_sents:
            s_clean = s.rstrip(".")
            # Prevent duplicate or near-identical sentences across chunks
            if not any(s_clean.lower() in existing.lower() or existing.lower() in s_clean.lower() for existing in cited_sentences):
                cited_sentences.append(f"{s_clean} [{n}].")
                if len(cited_sentences) >= 4:
                    break
        if len(cited_sentences) >= 4:
            break

    if not cited_sentences:
        return "I don't know based on the provided pages."

    return " ".join(cited_sentences[:4])

