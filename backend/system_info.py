"""
System info access (currently: the server's date/time).

The bug this fixes: the AI model was being asked "what's today's date?"
and, having no real clock, it guessed — and guessed wrong (a screenshot
showed it confidently stating "Monday, October 27, 2024", which was
simply false). A language model has no built-in notion of "now"; asking
it a clock question and trusting the answer is asking it to hallucinate
by design.

The fix is NOT "make the model smarter" — no amount of prompting fixes
a model not having a clock. The fix is: detect this specific class of
question, answer it with a real Python datetime.now() call instead of
an LLM call, and — per the explicit requirement — never do so without
the user's explicit, per-account, off-by-default permission first.

Nothing here is answered by Ollama. This is a deterministic code path,
same spirit as the existing stats-question fast path in rag.py.
"""

import re
from datetime import datetime
from backend.database import get_connection

_DATE_TIME_PATTERNS = [
    re.compile(r"\bwhat'?s (today'?s|the) (date|day)\b", re.I),
    re.compile(r"\bwhat is (today'?s|the) (date|day)\b", re.I),
    re.compile(r"\btoday'?s date\b", re.I),
    re.compile(r"\bcurrent date\b", re.I),
    re.compile(r"\bwhat day is it\b", re.I),
    re.compile(r"\bwhat'?s the time\b", re.I),
    re.compile(r"\bwhat is the time\b", re.I),
    re.compile(r"\bwhat time is it\b", re.I),
    re.compile(r"\bcurrent time\b", re.I),
    re.compile(r"\bdate and time\b", re.I),
    re.compile(r"\btime and date\b", re.I),
]


def is_system_info_question(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _DATE_TIME_PATTERNS)


def get_permission(username: str) -> bool:
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT allow_system_info FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    connection.close()
    return bool(row and row[0])


def set_permission(username: str, allowed: bool) -> None:
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE users SET allow_system_info = ? WHERE username = ?",
        (1 if allowed else 0, username),
    )
    connection.commit()
    connection.close()


PERMISSION_PROMPT = (
    "I can tell you the real current date and time, but I'd need your permission "
    "to check this system's clock first \u2014 I don't have that access by default. "
    "Would you like to allow it?"
)


def answer_system_info(text: str) -> str:
    """Deterministic answer using the server's actual clock. Only ever
    called after get_permission() has confirmed the user said yes."""
    now = datetime.now()
    date_str = now.strftime("%A, %B %-d, %Y")
    time_str = now.strftime("%-I:%M %p")

    wants_date = re.search(r"\bdate|day\b", text, re.I)
    wants_time = re.search(r"\btime\b", text, re.I)

    if wants_date and not wants_time:
        return f"Today is {date_str}."
    if wants_time and not wants_date:
        return f"It's currently {time_str} (system time)."
    return f"It's currently {time_str} on {date_str} (system time)."
