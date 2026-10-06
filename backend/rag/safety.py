"""
backend/rag/safety.py
Health Safety & Emergency Keywords Guard Module for HealthNest.
Protects users during medical emergencies or self-harm crises.

Rules:
1. Emergency keywords match: Return Emergency Banner alert.
2. Self-harm keywords match: Return supportive helpline guidance & SKIP generation.
3. Show medical disclaimer on every answer.
"""

from typing import Dict, Any, Tuple
import config
from backend.ir.text import normalize_text


def check_safety_guards(query: str) -> Dict[str, Any]:
    """
    Evaluates input query against emergency and self-harm keywords in config.py.
    
    Returns:
        {
            "is_emergency": bool,
            "is_self_harm": bool,
            "emergency_banner": str or None,
            "self_harm_message": str or None
        }
    """
    norm_q = normalize_text(query)

    # 1. Check self-harm keywords
    for keyword in config.SELF_HARM_KEYWORDS:
        if keyword in norm_q:
            return {
                "is_emergency": True,
                "is_self_harm": True,
                "emergency_banner": None,
                "self_harm_message": config.SELF_HARM_SAFETY_MESSAGE
            }

    # 2. Check general emergency keywords
    for keyword in config.EMERGENCY_KEYWORDS:
        if keyword in norm_q:
            return {
                "is_emergency": True,
                "is_self_harm": False,
                "emergency_banner": config.EMERGENCY_BANNER_TEXT,
                "self_harm_message": None
            }

    return {
        "is_emergency": False,
        "is_self_harm": False,
        "emergency_banner": None,
        "self_harm_message": None
    }
