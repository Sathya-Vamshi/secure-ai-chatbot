"""
Intelligent input sanitization + sanitized AI answers.

Replaces the old all-or-nothing `detect_prompt_injection()` behavior
(one regex hit -> whole request blocked) with a layer that:

  1. DETECTS specific categories of unsafe instruction (prompt-injection,
     auth/integrity bypass attempts, system-prompt/secret extraction,
     RAG manipulation) at the level of individual clauses, not the
     whole message.
  2. SEPARATES the unsafe clause(s) from any legitimate question sitting
     alongside them.
  3. SANITIZES: strips only the unsafe clause(s), keeps the rest, and
     hands the cleaned request on to the normal pipeline.
  4. Only refuses the ENTIRE request when nothing legitimate is left
     over after stripping — i.e. the request was unsafe start to finish.

This is deliberately NOT a single `.replace("bad phrase", "")` call and
NOT a "delete keywords, send the rest" filter: each category has its
own pattern family, some intentionally "tight" (matches only the
imperative instruction itself, e.g. "ignore document integrity checks")
and some intentionally "wide" (matches the whole ask, e.g. "reveal ...
system prompt", where anything joined to it by "and" is still part of
the same illegitimate request, not a separate legitimate one). That
distinction is what lets TEST 4 (bypass + legitimate question, same
sentence) and TEST 3 (pure extraction, nothing legitimate) both resolve
correctly instead of collapsing to the same behavior.

This module never grants access by itself — see NOTE in sanitize_input's
docstring. It only decides what text reaches the model and what part of
the model's answer reaches the user.
"""

import re

# ---------------------------------------------------------------------
# INPUT categories
# ---------------------------------------------------------------------

# "Tight" patterns: match only the imperative override/bypass clause
# itself, deliberately bounded so they don't reach across "and" into an
# unrelated, legitimate second clause in the same sentence.
_TIGHT_INPUT_PATTERNS = {
    "INJECTION_OVERRIDE": re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b(?:\s+\w+){0,4}?\s+"
        r"(?:instructions?|rules?|prompts?|guidelines?|restrictions?|limitations?|"
        r"safeguards?|filters?|constraints?|safety\s+(?:rules?|measures?|guidelines?))\b",
        re.I,
    ),
    "AUTH_BYPASS": re.compile(
        r"\b(?:ignore|bypass|skip|disable|override)\b(?:\s+\w+){0,3}?\s+"
        r"(?:access\s+control|authorization|authentication|permissions?|rbac)s?\b",
        re.I,
    ),
    "INTEGRITY_BYPASS": re.compile(
        r"\b(?:ignore|bypass|skip|disable|override)\b(?:\s+\w+){0,3}?\s+"
        r"(?:document\s+)?(?:integrity|tamper(?:ing)?|hash|sha-?256)\b"
        r"(?:\s+checks?|\s+verification)?",
        re.I,
    ),
    "RAG_MANIPULATION": re.compile(
        r"\b(?:ignore|bypass)\b(?:\s+\w+){0,3}?\s+(?:the\s+)?(?:retrieved|document)\s+context\b"
        r"|pretend\s+the\s+document\s+says"
        r"|use\s+your\s+own\s+knowledge\s+instead\s+of\s+the\s+document",
        re.I,
    ),
}

# "Wide" patterns: the whole ask is the violation, so a bounded gap
# (`.{0,60}`) is intentional — anything joined to it (e.g. "... and
# security instructions") is still part of the same illegitimate ask,
# not a separate salvageable question.
_WIDE_INPUT_PATTERNS = {
    "SYSTEM_PROMPT_LEAK": re.compile(
        r"\b(?:reveal|show|display|print|expose|give me|tell me|share)\b.{0,60}\b"
        r"(?:system prompt|hidden instructions|hidden rules|internal instructions|"
        r"internal rules|security instructions|security rules|prompt template)\b",
        re.I,
    ),
    "SECRET_REQUEST": re.compile(
        r"\b(?:reveal|show|display|give me|tell me|share|provide)\b.{0,60}\b"
        r"(?:secret key|password|api key|credentials?|access token|"
        r"admin(?:istrator)?'?s?\s+(?:authorization\s+)?key|authorization key)\b",
        re.I,
    ),
    "UNAUTHORIZED_INFO_REQUEST": re.compile(
        r"\b(?:give me|show me|provide|share)\b.{0,60}\b"
        r"(?:customer'?s?|admin(?:istrator)?'?s?|user'?s?|employee'?s?)\s+"
        r"(?:private|confidential|personal|secret)\s+information\b",
        re.I,
    ),
}

_ALL_INPUT_PATTERNS = {**_TIGHT_INPUT_PATTERNS, **_WIDE_INPUT_PATTERNS}

# Stray connector words/punctuation left behind after a clause is
# stripped out ("... and explain SHA-256" -> "explain SHA-256").
_LEADING_CONNECTOR_RE = re.compile(
    r"^\s*(?:,|;|and|also|additionally|furthermore|then|please|kindly|just|well|ok|okay|so|now)\b[\s,]*",
    re.I,
)
_TRAILING_CONNECTOR_RE = re.compile(
    r"[\s,]*\b(?:and|also|additionally|furthermore|then)\s*$", re.I
)

# Filler words that don't count as "legitimate content" on their own —
# used to stop a leftover like "Please and." from being mistaken for a
# salvaged question just because it has >=3 letters.
_FILLER_WORDS = {
    "please", "kindly", "just", "well", "ok", "okay", "so", "now", "and",
    "also", "then", "additionally", "furthermore", "the", "a", "an", "to",
}


def _clean_clause(text: str) -> str:
    # Leading filler can appear more than once ("Please and ignore..."
    # after a removal might leave "Please and X") — strip repeatedly.
    prev = None
    while prev != text:
        prev = text
        text = _LEADING_CONNECTOR_RE.sub("", text)
    text = _TRAILING_CONNECTOR_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,;")


def _has_meaningful_content(text: str) -> bool:
    """After stripping unsafe clauses and connector/filler debris, is
    there still an actual question/request left, or just scraps?"""
    words = re.findall(r"[a-zA-Z]+", text)
    significant = [w for w in words if w.lower() not in _FILLER_WORDS and len(w) >= 3]
    return len(significant) >= 1


def sanitize_input(text: str) -> dict:
    """
    NOTE ON SCOPE: this function only decides what text is safe to hand
    to the model. It never checks or grants permissions — authentication,
    role checks, and document-ownership/authorization rules in main.py
    and authorization.py run completely independently, before and after
    this, and are not affected by anything this function decides.

    Returns:
      {
        "risk_level": "low" | "medium" | "high",
        "sanitized_text": str or None,   # None only when risk_level == "high"
        "input_sanitized": bool,
        "categories": [str, ...],        # which categories fired, for logging
      }
    """
    if not text or not text.strip():
        return {"risk_level": "low", "sanitized_text": text, "input_sanitized": False, "categories": []}

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    kept_parts = []
    fired_categories = set()

    for sentence in sentences:
        cleaned = sentence
        for category, pattern in _ALL_INPUT_PATTERNS.items():
            new_cleaned, count = pattern.subn("", cleaned)
            if count > 0:
                fired_categories.add(category)
                cleaned = new_cleaned

        cleaned = _clean_clause(cleaned)

        if _has_meaningful_content(cleaned):
            kept_parts.append(cleaned)
        # else: this sentence contributed nothing safe — dropped entirely.

    if not fired_categories:
        return {"risk_level": "low", "sanitized_text": text, "input_sanitized": False, "categories": []}

    safe_text = ". ".join(p.rstrip(". ") for p in kept_parts if p).strip()
    if safe_text:
        safe_text = safe_text[0].upper() + safe_text[1:]
        if not safe_text.endswith((".", "!", "?")):
            safe_text += "."

    if not safe_text or not _has_meaningful_content(safe_text):
        return {
            "risk_level": "high", "sanitized_text": None,
            "input_sanitized": True, "categories": sorted(fired_categories),
        }

    return {
        "risk_level": "medium", "sanitized_text": safe_text,
        "input_sanitized": True, "categories": sorted(fired_categories),
    }


def refusal_message_for(categories: list) -> str:
    """A clear, safe explanation for a HIGH-risk request — not a bare
    'request rejected', per the spec. Deliberately generic about which
    exact phrase tripped it (no detection internals exposed)."""
    return (
        "I can't act on that request as written — it's asking me to bypass "
        "security controls (like ignoring my instructions, revealing internal "
        "system details, or accessing information you're not authorized for), "
        "and there's no other question in there for me to answer. If you have "
        "a genuine question about the document or how the system works, feel "
        "free to ask it directly and I'm glad to help."
    )


# ---------------------------------------------------------------------
# OUTPUT sanitization
# ---------------------------------------------------------------------

# Strong signal that the model echoed back internal prompt scaffolding
# verbatim (our own template markers) — this is severe enough that we
# don't try to salvage partial content around it.
_INTERNAL_TEMPLATE_MARKERS = re.compile(
    r"DOCUMENT EXCERPT:|USER MESSAGE:|You are a helpful, friendly AI assistant built into a secure",
    re.I,
)

# Redact-in-place patterns: remove just the leaked value/clause, keep
# the rest of the sentence, matching the spec's own example exactly
# (TEST 6: keep the SHA-256 sentence, redact only the key value).
_OUTPUT_REDACT_PATTERNS = [
    (
        re.compile(
            r"\b((?:the\s+)?(?:secret\s+)?(?:admin(?:istrator)?'?s?\s+)?"
            r"(?:authorization\s+key|api\s*key|access\s+token|password|secret\s+key|credentials?))"
            r"\s+(?:is|are)\s*:?\s*[\"']?[A-Za-z0-9\-_@#$%^&*.]{3,}[\"']?",
            re.I,
        ),
        lambda m: f"{m.group(1)[0].upper()}{m.group(1)[1:]} cannot be disclosed.",
    ),
]


def sanitize_output(raw_answer: str) -> dict:
    """
    Returns:
      {
        "sanitized_text": str,
        "output_sanitized": bool,
        "categories": [str, ...],
        "blocked_entirely": bool,
      }
    """
    if not raw_answer or not raw_answer.strip():
        return {"sanitized_text": raw_answer, "output_sanitized": False, "categories": [], "blocked_entirely": False}

    if _INTERNAL_TEMPLATE_MARKERS.search(raw_answer):
        return {
            "sanitized_text": (
                "I can't share that response as generated, since it included internal "
                "system details that shouldn't be exposed. Could you rephrase your "
                "question and I'll try again?"
            ),
            "output_sanitized": True,
            "categories": ["SYSTEM_PROMPT_LEAK"],
            "blocked_entirely": True,
        }

    cleaned = raw_answer
    fired = []
    for pattern, replacer in _OUTPUT_REDACT_PATTERNS:
        new_cleaned, count = pattern.subn(replacer, cleaned)
        if count > 0:
            fired.append("SECRET_DISCLOSURE")
            cleaned = new_cleaned

    if not fired:
        return {"sanitized_text": raw_answer, "output_sanitized": False, "categories": [], "blocked_entirely": False}

    return {"sanitized_text": cleaned, "output_sanitized": True, "categories": sorted(set(fired)), "blocked_entirely": False}
