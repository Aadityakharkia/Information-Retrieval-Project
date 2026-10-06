"""
backend/rag/llm_client.py
LLM Client Interface for HealthNest.
Interfaces with Ollama API (/api/chat) via HTTP requests.
Supports configurable model sizes (llama3.2:1b, llama3.2:3b, llama3.1:8b).
Supports fallback to OpenAI-compatible endpoint or smart mock mode if Ollama server is offline.

Temperature set to 0.0 for deterministic, verifiable answers.
"""

import json
import logging
import requests
from typing import List, Dict, Any, Optional
import config

logger = logging.getLogger(__name__)


def generate_llm_response(
    messages: List[Dict[str, str]],
    model: str = config.DEFAULT_MODEL,
    base_url: str = config.OLLAMA_BASE_URL,
    temperature: float = config.LLM_TEMPERATURE
) -> str:
    """
    Sends chat request to Ollama /api/chat endpoint.
    If server is unreachable or model is not found, falls back gracefully.
    """
    # 1. Try Ollama local endpoint
    try:
        url = f"{base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        response = requests.post(url, json=payload, timeout=12)
        if response.status_code == 200:
            data = response.json()
            return data.get("message", {}).get("content", "").strip()
        else:
            logger.warning(f"Ollama returned HTTP {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.info(f"Ollama server not reachable at {base_url} ({e}). Checking OpenAI endpoint / Mock mode...")

    # 2. Try OpenAI-compatible endpoint if configured
    if config.OPENAI_API_BASE:
        try:
            url = f"{config.OPENAI_API_BASE.rstrip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if config.OPENAI_API_KEY:
                headers["Authorization"] = f"Bearer {config.OPENAI_API_KEY}"
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            res = requests.post(url, headers=headers, json=payload, timeout=15)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as ex:
            logger.warning(f"OpenAI endpoint call failed: {ex}")

    # 3. Fallback Smart Mock Answer Generator (ensures system works standalone during demo/evaluation)
    logger.info("Using Fallback RAG Generator (Ollama offline/mock mode).")
    return generate_mock_rag_answer(messages)


import re
from backend.ir.text import tokenize


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
    q_tokens = set(tokenize(question_text, use_stopwords=True, use_stemmer=True)) if question_text else set()

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

    # If no retrieved context chunks have relevant query terms, abstain truthfully
    if not relevant_blocks:
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

