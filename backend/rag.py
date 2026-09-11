"""
Lightweight RAG (retrieval) + document-stats helpers.

Goal: stop sending the WHOLE document to the LLM on every question.

Two things happen here:

1. STATS QUESTIONS ("how many words / letters / numbers are in the
   document?") are answered with plain Python, instantly, with zero
   LLM calls. The old code failed on these because the literal answer
   ("this document has 4,213 words") is never *written* in the
   document text, so the model had nothing to point to.

2. CONTENT QUESTIONS ("what is the password policy?") are answered by
   splitting the document into chunks, scoring each chunk against the
   question with simple keyword overlap (TF-style, no embedding model
   needed), and sending only the top few chunks to the LLM instead of
   the entire document. This is what collapses a 4-5 minute prompt
   into a few-hundred-word one.

No extra dependencies (no sentence-transformers / faiss / sklearn).
This keeps setup simple; it can be swapped for real vector embeddings
later without changing the calling code in main.py.
"""

import re
from collections import Counter

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to",
    "for", "and", "or", "what", "how", "many", "do", "does", "did", "i",
    "it", "this", "that", "with", "as", "at", "be", "by", "from", "have",
    "has", "you", "me", "please", "tell", "about", "can", "could", "would",
    "there", "which", "who", "whom", "my", "our", "we", "will",
}

# NOTE: each category allows an optional trailing "s" (letter/letters,
# word/words, ...). A question like "how many letter and words are
# there" uses the singular "letter" — without the "s?" this whole
# question would silently fail to match and fall through to the LLM,
# which has no way to actually count anything in the text.
_STAT_WORD = r"(words?|letters?|characters?|numbers?|digits?|lines?|sentences?|paragraphs?)"

STATS_PATTERNS = [
    rf"\bhow many\b.{{0,25}}\b{_STAT_WORD}\b",
    r"\bword count\b",
    r"\bcharacter count\b",
    r"\bletter count\b",
    rf"\bnumber of\b.{{0,25}}\b{_STAT_WORD}\b",
    rf"\btotal\b.{{0,25}}\b{_STAT_WORD}\b",
]


def is_stats_question(question: str) -> bool:
    """True for meta questions about the document itself (counts), which
    have no LLM-writable answer and should be computed directly instead."""
    q = question.lower()
    return any(re.search(p, q) for p in STATS_PATTERNS)


def compute_document_stats(text: str) -> dict:
    words = re.findall(r"\b[\w'-]+\b", text)
    letters = [c for c in text if c.isalpha()]
    digits = [c for c in text if c.isdigit()]
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip() != ""]

    return {
        "characters_total": len(text),
        "letters": len(letters),
        "digits": len(digits),
        "words": len(words),
        "lines": len(lines),
        "sentences": len(sentences),
    }


def answer_stats_question(question: str, text: str) -> str:
    """Answers EVERY stat category mentioned in the question, not just the
    first one matched. "how many letters and words are there" used to
    silently return only the word count — each category below is checked
    independently instead of an if/elif chain that stops at the first hit."""
    stats = compute_document_stats(text)
    q = question.lower()

    found = []
    if "word" in q:
        found.append(f"{stats['words']} words")
    if "letter" in q:
        found.append(f"{stats['letters']} letters")
    if "character" in q:
        found.append(f"{stats['characters_total']} characters")
    if "number" in q or "digit" in q:
        found.append(f"{stats['digits']} numeric digits")
    if "line" in q:
        found.append(f"{stats['lines']} non-empty lines")
    if "sentence" in q:
        found.append(f"{stats['sentences']} sentences")
    if "paragraph" in q:
        paragraphs = [p for p in text.split("\n\n") if p.strip() != ""]
        found.append(f"{len(paragraphs)} paragraphs")

    if not found:
        # Fallback: dump everything we know.
        return (
            f"Document stats — words: {stats['words']}, letters: {stats['letters']}, "
            f"digits: {stats['digits']}, characters: {stats['characters_total']}, "
            f"lines: {stats['lines']}, sentences: {stats['sentences']}."
        )

    if len(found) == 1:
        return f"The document contains {found[0]}."

    return "The document contains " + ", ".join(found[:-1]) + f", and {found[-1]}."


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    """Split into overlapping word-based chunks so an answer that spans a
    chunk boundary isn't lost."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def retrieve_relevant_chunks(text: str, question: str, top_k: int = 3,
                              chunk_size: int = 700, overlap: int = 100) -> str:
    """Return the top_k chunks most relevant to the question, concatenated.
    Falls back to the whole (short) document if it's small enough that
    chunking wouldn't help."""
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    if len(chunks) <= 1:
        return text

    q_tokens = Counter(_tokenize(question))
    if not q_tokens:
        # Nothing meaningful to match on — just take the first few chunks.
        return "\n\n---\n\n".join(chunks[:top_k])

    scored = []
    for chunk in chunks:
        c_tokens = Counter(_tokenize(chunk))
        score = sum(min(count, c_tokens[word]) for word, count in q_tokens.items())
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [c for score, c in scored[:top_k] if score > 0]

    if not top_chunks:
        # No keyword overlap at all — the question may use different
        # wording than the document. Send the first couple of chunks
        # rather than nothing, so the model can still say "not found".
        top_chunks = [c for _, c in scored[:top_k]]

    return "\n\n---\n\n".join(top_chunks)
