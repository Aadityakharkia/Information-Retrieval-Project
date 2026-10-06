"""
backend/rag/prompt.py
Prompt Engineering & Context Formatting Module for HealthNest.
Constructs strict grounding prompt templates for the LLM.

Strict Rules Enforced:
1. Answer ONLY from provided context pages [1]..[K].
2. Write one clear factual sentence per line or claim.
3. Every sentence MUST end with its source citation marker like [1].
4. If context is missing or insufficient, reply EXACTLY: "I don't know based on the provided pages."
"""

from typing import List, Dict, Any

SYSTEM_PROMPT = """You are HealthNest, an expert evidence-grounded medical Q&A assistant.
Your absolute duty is to provide strictly trustworthy answers using ONLY the provided reference pages.

STRICT GENERATION RULES:
1. Answer ONLY using facts directly stated in the reference context pages below.
2. Do NOT use outside medical knowledge, general assumptions, or extrapolate.
3. Every single sentence MUST express a single clear fact and end with a citation marker like [1] or [2] corresponding to the reference page used.
4. If the provided reference pages do NOT contain enough information to answer the question, reply EXACTLY with:
"I don't know based on the provided pages."
"""


def format_rag_prompt(question: str, top_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Formats system prompt and user context message for LLM chat API.
    """
    context_str_parts = []
    for idx, item in enumerate(top_chunks, start=1):
        chunk = item.get("chunk", item)
        q_text = chunk.get("question", "")
        body_text = chunk.get("text", "")
        context_str_parts.append(
            f"[{idx}] Reference Page {idx} (ID: {chunk.get('chunk_id', 'unknown')})\n"
            f"Question: {q_text}\n"
            f"Content: {body_text}\n"
        )

    context_text = "\n".join(context_str_parts)

    user_content = (
        f"Context Pages:\n{context_text}\n\n"
        f"User Health Question: {question}\n\n"
        f"Instructions: Write a concise, factual answer where EVERY sentence ends with a citation [n]. "
        f"If context is insufficient, output EXACTLY 'I don't know based on the provided pages.'"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
