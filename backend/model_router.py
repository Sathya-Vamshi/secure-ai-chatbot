"""
Model routing.

Decides two things:
  1. How hard is this question? ("trivial" / "simple" / "complex")
  2. Given the fast model's answer, does it look weak enough that we
     should silently retry with the deep model before replying?

Why this exists (upgrade #1 from the roadmap: "answer minimal question to
hard question also"):

The old routing was a single boolean (`needs_deep_model`) based on a
length check and a short keyword list. That has two problems:

  - Trivial chit-chat ("hi", "thanks") still went through the full
    RAG + fast-model pipeline for no reason.
  - A genuinely hard question that didn't happen to contain one of the
    magic keywords ("why", "compare", ...) got sent to the small model
    and there was no safety net if the answer came back weak/short.

This module replaces that with:
  - classify_difficulty(): a small scoring function using several
    signals instead of one keyword list.
  - trivial_reply(): canned, instant answers for greetings/thanks/bye
    so those never touch an LLM at all.
  - is_weak_answer(): a check run AFTER the fast model answers a
    "simple" question; if it looks like the model under-delivered,
    main.py retries once with the deep model instead of returning a
    weak answer as final.

Still 100% offline, no extra dependencies — just clearer rules.
"""

import re

# ---------------------------------------------------------------------
# Trivial chit-chat: answered instantly, no LLM call at all.
# ---------------------------------------------------------------------

_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|yo|hiya|sup)[\s!.,]*$", re.I)
_MORNING_RE = re.compile(r"^\s*good\s+(morning|afternoon|evening|night)[\s!.,]*$", re.I)
_THANKS_RE = re.compile(r"^\s*(thanks|thank you|thx|ty|appreciate it)[\s!.,]*$", re.I)
_BYE_RE = re.compile(r"^\s*(bye|goodbye|see ya|see you|cya|later)[\s!.,]*$", re.I)
_ACK_RE = re.compile(r"^\s*(ok|okay|k|cool|nice|great|got it|alright)[\s!.,]*$", re.I)

_TRIVIAL_PATTERNS = (_GREETING_RE, _MORNING_RE, _THANKS_RE, _BYE_RE, _ACK_RE)


def trivial_reply(question: str) -> str | None:
    """Return a canned reply for common small talk, or None if this isn't
    small talk (in which case it should go through the normal pipeline
    even though classify_difficulty() may still call it 'trivial')."""
    q = question.strip()

    if _GREETING_RE.match(q) or _MORNING_RE.match(q):
        return "Hello! Ask me anything — about a document you've uploaded, or anything else you'd like help with."
    if _THANKS_RE.match(q):
        return "You're welcome! Let me know if there's anything else I can help with."
    if _BYE_RE.match(q):
        return "Goodbye! Come back anytime."
    if _ACK_RE.match(q):
        return "Got it. What would you like to know?"

    return None


# ---------------------------------------------------------------------
# Difficulty scoring for real (non-chit-chat) questions.
# ---------------------------------------------------------------------

COMPLEX_KEYWORDS = [
    "explain in detail", "analyze", "analyse", "summarize the whole",
    "summarise the whole", "compare", "why", "pros and cons",
    "step by step", "step-by-step", "list all", "in depth", "in-depth",
    "difference between", "advantages and disadvantages", "evaluate",
    "recommend", "recommendation", "what should", "how should",
    "walk me through", "elaborate", "justify", "critique", "trade-off",
    "tradeoff", "root cause", "implications",
]


def classify_difficulty(question: str) -> str:
    """Returns 'trivial', 'simple', or 'complex'.

    'trivial' here just means "short enough to be chit-chat" — main.py
    still calls trivial_reply() to check whether it actually IS chit-chat
    before skipping the LLM entirely.
    """
    q = question.strip()

    if not q:
        return "trivial"

    if any(p.match(q) for p in _TRIVIAL_PATTERNS):
        return "trivial"

    ql = q.lower()
    score = 0

    score += len(q) // 120                                    # sheer length
    score += ql.count("?")                                     # multiple questions at once
    score += sum(1 for kw in COMPLEX_KEYWORDS if kw in ql)      # analytical language
    if len(re.findall(r"\band\b|\bor\b", ql)) >= 2:             # multi-part ask
        score += 1
    if len(q.split()) > 40:                                     # long question
        score += 1

    return "complex" if score >= 2 else "simple"


# ---------------------------------------------------------------------
# Weak-answer detection, used to decide whether to escalate a "simple"
# question's fast-model answer up to the deep model.
# ---------------------------------------------------------------------

_WEAK_ANSWER_RE = re.compile(
    r"^(i cannot find that information|i don'?t know|i'?m not sure|"
    r"unable to answer|sorry, i (can'?t|cannot))",
    re.I,
)


def is_weak_answer(answer: str, question: str) -> bool:
    """True if the fast model's answer looks like it under-delivered for
    this question, and a retry with the deep model is worth the extra
    time."""
    a = (answer or "").strip()

    if not a:
        return True

    if _WEAK_ANSWER_RE.match(a):
        # A flat "not in the document" is a perfectly fine, correct answer
        # for a plain lookup question. It's only worth a retry if the
        # question looked like it wanted real explanation/analysis.
        return classify_difficulty(question) == "complex"

    # An answer that's suspiciously short for a question that clearly
    # asked for detail/explanation.
    if classify_difficulty(question) == "complex" and len(a.split()) < 15:
        return True

    return False
