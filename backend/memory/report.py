"""
backend/memory/report.py
Reads medical test reports (blood tests, scans) from the patient's PDFs.

Lines are shown VERBATIM. The only thing calculated is whether a value lies inside the reference range
printed on the same line of the same report. Anything ambiguous (two ranges, two values, unclear
units, words instead of numbers) is left without a comparison rather than guessed.
"""

import re
from typing import Any, Dict, List, Optional

from backend.ir.text import normalize_text, tokenize

_NUM = r"\d+(?:,\d{2,3})*(?:\.\d+)?"
# a-b (not part of a date, time, phone number or longer dash chain)
_BETWEEN = re.compile(rf"(?<![\d.,:/-])({_NUM})\s*(?:-|–|—|\bto\b)\s*({_NUM})(?![\d:/]|\s*[-–—]\s*\d|[.,]\d)", re.I)
_LIMIT = re.compile(rf"(<=|>=|≤|≥|<|>|\bup\s*to\b|\bless than\b|\bmore than\b|\bgreater than\b)\s*({_NUM})", re.I)
_VALUE = re.compile(rf"(?<!\S)\*?({_NUM})\*?(?!\S)")
_FLAG = re.compile(r"(?<!\S)\(?(H|L|High|Low)\)?(?!\S)", re.I)
_NOT_A_TEST = re.compile(r"\b(age|date|phone|mobile|tel|page|patient|sample|collected|reported|received|printed|time|bill|reg|registration|id|no|"
                         r"address|pin|doctor|dr|ref(?:erred)? by|test|result)\b", re.I)
_SCALED_UNIT = re.compile(r"lakh|lac\b|million|thousand|10\s*\^|10\*|x\s*10|×\s*10", re.I)
_DOSE = re.compile(r"\b(?:tab(?:let)?s?|cap(?:sule)?s?|syp|syrup|inj(?:ection)?)\b|\b(?:once|twice|thrice|daily|after food|before food)\b", re.I)


def _f(x: str) -> float:
    return float(x.replace(",", ""))


def parse_row(line: str) -> Optional[Dict[str, Any]]:
    """One table line -> {line, name, value, range, status}. status is None when no safe comparison is possible."""
    if _DOSE.search(line):
        return None
    ranges = list(_BETWEEN.finditer(line)) or list(_LIMIT.finditer(line))
    if not ranges:
        return None
    rng = ranges[0]
    before, after = line[:rng.start()], line[rng.end():]
    values = list(_VALUE.finditer(before))
    if not values:
        return None
    name = re.sub(r"[\s:.\-*]+$", "", before[:values[0].start()]).strip()
    if len(re.findall(r"[A-Za-z]", name)) < 2 or _NOT_A_TEST.search(name):
        return None
    row = {"line": line, "name": name, "value": values[0].group(1), "range": rng.group(0).strip(), "status": None}
    # exactly one value, one range, and no other number after the range
    if len(values) != 1 or len(ranges) != 1 or _VALUE.search(after) or _LIMIT.search(after):
        return row
    v = _f(values[0].group(1))
    if rng.re is _BETWEEN:
        if re.search(r"[<>≤≥]", line):  # "< 10 - 20": malformed, do not guess
            return row
        lo, hi = _f(rng.group(1)), _f(rng.group(2))
        if lo >= hi:
            return row
        status = "below" if v < lo else "above" if v > hi else "within"
    else:
        op, lim = rng.group(1).lower(), _f(rng.group(2))
        lo = hi = lim
        if op in ("<", "less than"):
            status = "within" if v < lim else "above"
        elif op in (">", "more than", "greater than"):
            status = "within" if v > lim else "below"
        elif op in (">=", "≥"):
            status = "within" if v >= lim else "below"
        else:  # <=, ≤, up to
            status = "within" if v <= lim else "above"
    # "2.1 lakh" against "150000 - 450000": value and range are in different scales
    if _SCALED_UNIT.search(line) and not (lo / 20 <= v <= hi * 20):
        return row
    # never contradict the laboratory's own High/Low mark
    flag = _FLAG.search(line[values[0].end():])
    if flag and {"h": "above", "l": "below"}[flag.group(1)[0].lower()] != status:
        return row
    row["status"] = status
    return row


def rows(text: str) -> List[Dict[str, Any]]:
    return [r for r in map(parse_row, text.splitlines()) if r]


_REPORT_TEXT = re.compile(r"\b(reference (?:range|interval|value)s?|biological ref|normal (?:range|value)s?|impression|findings|diagnostics?|patholog\w+|"
                          r"laborator\w+|specimen|ultrasound|usg|x-?ray|mri|ct scan|ecg|echo(?:cardiogra\w+)?|biopsy|blood count|lipid profile|imaging)\b", re.I)


def is_report(text: str) -> bool:
    return len(rows(text)) >= 2 or len({m.lower() for m in _REPORT_TEXT.findall(text)}) >= 2


def note(row: Dict[str, Any]) -> Optional[str]:
    if row["status"] is None:
        return None
    word = {"below": "Below", "above": "Above", "within": "Within"}[row["status"]]
    return f"{word} the range printed on your report ({row['range']})"


# Everyday words -> words that appear as test names on reports (English, Hinglish, Hindi)
ALIASES = {
    "sugar": "glucose sugar hba1c fbs rbs ppbs", "glucose": "glucose sugar fbs rbs ppbs", "diabetes": "glucose sugar hba1c", "शुगर": "glucose sugar hba1c",
    "a1c": "hba1c", "hba1c": "hba1c a1c", "hemoglobin": "hemoglobin haemoglobin hb", "haemoglobin": "hemoglobin haemoglobin hb", "hb": "hemoglobin haemoglobin hb",
    "khoon": "hemoglobin haemoglobin hb", "खून": "hemoglobin haemoglobin hb",
    "cholesterol": "cholesterol ldl hdl triglycerides", "lipid": "cholesterol ldl hdl triglycerides vldl", "कोलेस्ट्रॉल": "cholesterol ldl hdl triglycerides",
    "thyroid": "tsh t3 t4 ft3 ft4", "थायराइड": "tsh t3 t4", "kidney": "creatinine urea bun egfr", "gurda": "creatinine urea egfr",
    "liver": "sgot sgpt alt ast bilirubin alp liver", "platelets": "platelet", "platelet": "platelet", "wbc": "wbc leucocyte leukocyte",
    "vitamin": "vitamin", "uric": "uric",
}
_STOP = {"blood", "urin", "report", "result", "test", "level", "valu", "count", "total", "serum", "scan", "show", "normal", "abnorm", "high", "low", "lab", "explain", "mean",
         "kitna", "kitni", "kitn", "batao", "lower", "reduc", "improv", "say"}


def query_terms(question: str) -> set:
    words = re.findall(r"[\w\u0900-\u097F]+", normalize_text(question))
    terms = set()
    for w in words:
        for t in tokenize(w):
            if t not in _STOP and len(t) > 1:
                terms.add(t)
        for alias in ALIASES.get(w, "").split():
            terms.update(tokenize(alias))
    return terms


def matching_lines(question: str, text: str) -> List[str]:
    """Report lines that mention what the question is about, in document order."""
    terms = query_terms(question)
    lines = text.splitlines()
    parsed = {r["line"]: r for r in rows(text)}
    out = []
    for ln in lines:
        target = parsed[ln]["name"] if ln in parsed else ln
        if ln not in parsed and re.match(r"(?i)\s*(patient|name|age|date|address|test\s+result)", ln):
            continue
        if terms & set(tokenize(target)):
            out.append(ln)
    return out


def topic_of_row(row: Dict[str, Any]) -> str:
    """Searchable name of a test: 'Total Cholesterol' -> 'Cholesterol'."""
    name = re.sub(r"\(.*?\)", " ", row["name"])
    words = [w for w in re.findall(r"[A-Za-z][\w-]*", name) if w.lower() not in ("total", "serum", "count", "level", "fasting", "random", "plasma", "blood", "direct")]
    return " ".join(words) or row["name"]


_GENERIC_TITLE_WORDS = {"high", "low", "blood", "test", "tests", "level", "levels", "count", "deficiency", "a", "the"}


def page_is_about(test: str, page_topic: str) -> bool:
    """True only if a page is about the test itself ('high blood cholesterol' for Cholesterol).
    'hemoglobin e disease' for Hemoglobin is rejected: a result must never suggest an unrelated disease."""
    want = set(tokenize(test))
    words = [w for w in page_topic.split() if w not in _GENERIC_TITLE_WORDS]
    have = {t for w in words for t in tokenize(w)}
    return bool(have) and have <= want
