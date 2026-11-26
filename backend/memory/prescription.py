"""
backend/memory/prescription.py
Reads the patient's PDFs (prescriptions here, test reports in report.py) and picks what to show.

Prescription lines are always shown VERBATIM. They are never paraphrased by a language model and
never split into sentences, so a dose can not be altered on the way to the patient.
"""

import io
import re
import unicodedata
from typing import Any, Dict, List, Tuple

from pypdf import PdfReader

import config
from backend.ir.bm25 import BM25Retriever
from backend.ir.inverted_index import InvertedIndex
from backend.ir.text import normalize_text, tokenize
from backend.memory import report


class PdfError(ValueError):
    """Message is safe to show to the user."""


def _page_text(page) -> str:
    """
    Text of one page, read in LAYOUT mode: words are placed by their position, so each table row stays
    on one line ("Hemoglobin 11.40 g/dL 12.00 - 15.00"). Plain mode follows the PDF's drawing order,
    which for many lab reports (e.g. column layouts from large Indian labs) puts the method on its own
    line and glues range, unit and result together ("12.00 - 15.00 g/dL11.40", "mill/mm34.07"), which
    cannot be split safely. Falls back to plain mode if layout mode fails or finds nothing.
    """
    try:
        t = page.extract_text(extraction_mode="layout") or ""
    except Exception:
        t = ""
    return t if t.strip() else (page.extract_text() or "")


def extract_pdf(data: bytes) -> Tuple[str, int]:
    """Returns (text, page_count). Raises PdfError for anything that is not a readable text PDF."""
    if not data.startswith(b"%PDF-"):
        raise PdfError("This file is not a PDF. Please upload a PDF prescription or report.")
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise PdfError("This PDF is password protected. Please upload an unlocked copy.")
        pages = len(reader.pages)
        if pages > config.MAX_PDF_PAGES:
            raise PdfError(f"This PDF has {pages} pages (maximum {config.MAX_PDF_PAGES}).")
        parts, size = [], 0
        for page in reader.pages:
            t = _page_text(page)
            parts.append(t)
            size += len(t)
            if size > config.MAX_DOC_CHARS:
                raise PdfError("This PDF contains too much text.")
    except PdfError:
        raise
    except Exception as exc:  # malformed PDFs raise many different errors inside the parser
        raise PdfError("This PDF is damaged and could not be read.") from exc
    text = unicodedata.normalize("NFKC", "\n".join(parts)).replace("\x00", "")
    # Layout mode pads columns with runs of spaces; one space is enough once each row is on its own line.
    text = "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())
    if sum(ch.isalnum() for ch in text) < 20:
        raise PdfError("No readable text found. Scanned or handwritten documents are not supported yet.")
    return text, pages


_FORM = r"tab(?:let)?s?|cap(?:sule)?s?|syp|syrup|inj(?:ection)?|oint(?:ment)?|cream|gel|drops?|susp(?:ension)?|inhaler|lotion"
_FORM_FIRST = re.compile(rf"\b(?:{_FORM})\b[.:]?\s+([A-Za-z][A-Za-z-]{{2,}}(?:\s+[A-Z][A-Za-z-]{{2,}})?)", re.I)
_DOSE_AFTER = re.compile(r"\b([A-Za-z][A-Za-z-]{3,})\s+(?:(?:" + _FORM + r")\b[.:]?\s+)?\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ml|iu|g)\b(?!\s*/\s*(?:d|l\b|ml\b|cumm|24))", re.I)
_NOT_MEDICINE = set("""weight height pressure sugar glucose dose dosage total daily take after before food with water for days day
weeks week month morning night evening afternoon times time once twice thrice each every patient name age date doctor clinic hospital
apply use continue stop and the one two half strength quantity maximum minimum fasting random cholesterol creatinine hemoglobin
haemoglobin sodium potassium supplement supplements""".split())


def medicine_lines(text: str) -> List[Tuple[str, List[str]]]:
    """[(verbatim line, [medicine names found on it])] for lines that look like a medicine entry."""
    out = []
    for line in text.splitlines():
        if report.parse_row(line):
            continue  # a test result, not a medicine
        names = []
        for m in _FORM_FIRST.findall(line) + _DOSE_AFTER.findall(line):
            first = [w for w in m.split() if w.lower() not in _NOT_MEDICINE and not re.fullmatch(_FORM, w, re.I)]
            if first and first[0].lower() not in (n.lower() for n in names):
                names.append(first[0])
        if names:
            out.append((line, names))
    return out


def medicines(text: str) -> List[str]:
    seen: Dict[str, str] = {}
    for _, names in medicine_lines(text):
        for n in names:
            seen.setdefault(n.lower(), n)
    return list(seen.values())


_ME = re.compile(r"\b(my|mine|i|i'm|am i|meri|mera|mere|mujhe|meko|apni|apna)\b|मेरी|मेरा|मेरे|मुझे", re.I)
# The question is about the patient's own prescription ...
_PERSONAL = re.compile(
    r"\b(my|mine|me|i am taking|i'm taking|i take)\b.*\b(prescription|prescribed|medicines?|medications?|tablets?|pills?|doctor|dose|dosage|drugs?)\b"
    r"|\b(prescription|prescribed|doctor)\b.*\b(me|my|i|say|said|give|gave|given|write|wrote|advise|advised)\b"
    r"|\b(meri|mera|mere|mujhe|meko)\b.*\b(dawai|dawa|dawaiyan|goli|goliyan|parcha|parchi|prescription|doctor)\b"
    r"|(मेरी|मेरा|मेरे|मुझे).*(दवा|दवाई|दवाइयाँ|पर्चा|पर्ची|डॉक्टर)"
    r"|\bmy prescription\b|\bthis prescription\b", re.I)
# ... or their own test report.
_REPORT = re.compile(
    r"\b(my|mine|meri|mera|mere|apni|apna|this|uploaded)\b.*\b(reports?|results?|tests?|scans?|x-?ray|mri|ultrasound|sonography|usg|labs?)\b"
    r"|(मेरी|मेरा|मेरे).*(रिपोर्ट|जाँच|जांच|टेस्ट)", re.I)
_ABNORMAL = re.compile(r"\b(abnormal|out of range|not normal|bad|wrong|problem|kharab|gadbad|high or low|too high|too low)\b|खराब|गड़बड़", re.I)
# ... and asks for general knowledge, not just what is written.
_KNOWLEDGE = re.compile(
    r"side[ -]?effects?|\bwhy\b|what (?:is|are|does)\b.*\bfor\b|used for|\buses?\b|interact|safe|avoid|precaution|warning|how does|overdose|alcohol|pregnan|"
    r"\bkyun?\b|\bkis ?liye\b|nuksan|fayda|क्यों|किसलिए|नुकसान|दुष्प्रभाव", re.I)
_REPORT_KNOWLEDGE = re.compile(r"\b(why|means?|meaning|explain|causes?|lower|reduce|raise|increase|improve|control|treat(?:ment)?|diet|matlab|samjha\w*|kyun?|"
                               r"kaise (?:kam|badh\w*|theek))\b|मतलब|क्यों|कैसे", re.I)
MAX_LINES = 60


def analyse(question: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Decides whether a question involves the patient's documents, and picks the verbatim lines to show.
    sources: [{doc_id, filename, kind, lines: [{text, note}], hidden}]; topics: what to look up in the database.
    """
    result = {"used": False, "knowledge": False, "topics": [], "topic_kind": "medicine", "sources": []}
    if not docs:
        return result
    q_words = set(re.findall(r"\w+", normalize_text(question)))
    reports = [d for d in docs if report.is_report(d["text"])]
    scripts = [d for d in docs if d not in reports]
    # --- prescriptions (medicines may also be listed on a report) ---
    all_meds = {m.lower(): m for d in docs for m in medicines(d["text"])}
    named = [m for low, m in all_meds.items() if low in q_words]
    asks_rx = bool(_PERSONAL.search(question))
    asks_report = bool(_REPORT.search(question))
    if asks_rx and not scripts and not all_meds:
        asks_rx, asks_report = False, bool(reports)  # "what did my doctor say" with only reports uploaded
    if named or (asks_rx and (scripts or all_meds)):
        wanted = {m.lower() for m in named} or set(all_meds)
        for d in (docs if named else scripts or docs):
            med = medicine_lines(d["text"])
            if named:
                lines = [ln for ln, names in med if any(n.lower() in wanted for n in names)]
            else:
                lines = [ln for ln, _ in med]
                lines += [ln for ln in _matching_lines(question, d["text"]) if ln not in lines]
                lines = lines or d["text"].splitlines()
            _add(result, d, "report" if d in reports else "prescription", [(ln, None) for ln in lines])
        if _KNOWLEDGE.search(question) or (named and not asks_rx):
            result["topics"] += [all_meds[w] for w in all_meds if w in wanted]
            result["knowledge"] = True
        result["used"] = bool(result["sources"])
        if result["used"]:
            return result
    # --- test reports ---
    mine = bool(_ME.search(question))
    topics: List[str] = []
    specific = bool(report.query_terms(question))
    any_match = (mine or asks_report) and any(report.matching_lines(question, d["text"]) for d in reports)
    if asks_report and specific and reports and not any_match:
        result.setdefault("remarks", []).append("No line in your reports clearly matches this question, so the whole report is shown.")
    for d in reports:
        rows_ = {r["line"]: r for r in report.rows(d["text"])}
        about = report.matching_lines(question, d["text"]) if (mine or asks_report) else []
        if about:
            lines = about
        elif asks_report and not any_match:
            lines = d["text"].splitlines()
        else:
            continue
        # Only comparable values can be called out of range; say so instead of implying the rest is fine.
        if _ABNORMAL.search(question) and any(r["status"] for r in rows_.values()):
            out = [ln for ln in lines if ln in rows_ and rows_[ln]["status"] in ("above", "below")]
            unknown = [ln for ln in lines if ln not in rows_ or rows_[ln]["status"] is None]
            result.setdefault("remarks", []).append(
                f"{d['filename']}: {len(out)} value(s) outside the printed range. {len(unknown)} other line(s) could not be compared, so check the whole report.")
            lines = out
            if not lines:
                result["used"] = True
        pairs = [(ln, report.note(rows_[ln]) if ln in rows_ else None) for ln in lines]
        _add(result, d, "report", pairs)
        picked = [rows_[ln] for ln in lines if ln in rows_]
        if not about:  # whole report: explain only what is out of range
            picked = [r for r in picked if r["status"] in ("above", "below")]
        topics += [report.topic_of_row(r) for r in picked]
    if result["sources"]:
        result["used"] = True
        if _REPORT_KNOWLEDGE.search(question):
            result["topics"], result["knowledge"] = list(dict.fromkeys(topics))[:3], bool(topics)
            result["topic_kind"] = "test"
    return result


def _add(result: Dict[str, Any], doc: Dict[str, Any], kind: str, pairs: List[Tuple[str, Any]]) -> None:
    if pairs:
        result["sources"].append({"doc_id": doc["id"], "filename": doc["filename"], "kind": kind, "hidden": max(0, len(pairs) - MAX_LINES),
                                  "lines": [{"text": t, "note": n} for t, n in pairs[:MAX_LINES]]})


def _matching_lines(question: str, text: str, k: int = 4) -> List[str]:
    lines = text.splitlines()
    chunks = [{"chunk_id": str(i), "qa_id": "p", "question": "", "text": ln, "position": i} for i, ln in enumerate(lines)]
    index = InvertedIndex()
    index.index_chunks(chunks)
    stop = {t for w in ("prescription", "prescribed", "doctor", "medicine", "medication", "tablet", "report") for t in tokenize(w)}
    q = " ".join(t for t in re.findall(r"\w+", question) if not (set(tokenize(t)) & stop))
    return [r["chunk"]["text"] for r in BM25Retriever(index).search(q, top_k=k)]


_FILLER = re.compile(r"\b(my|mine|me|i|am|taking|take|prescription|prescribed|medicines?|medications?|tablets?|pills?|doctor|meri|mera|mere|mujhe|dawai|dawa|"
                     r"goli|ka|ki|ke|kya|hai|hain|these|this|the|of|are|is|do|does|for|what|which|can|should)\b", re.I)


def lookup_query(question: str, medicine: str) -> str:
    """General-knowledge query about one medicine, with personal words removed ('side effects Metformin')."""
    rest = " ".join(re.sub(r"[^\w\s-]", " ", _FILLER.sub(" ", question)).split())
    rest = " ".join(w for w in rest.split() if w.lower() != medicine.lower())
    return f"{medicine} {rest}".strip()
