# Resuming the project on your Mac

This does NOT start over. It picks up exactly where the status doc left off:
auth, upload, SHA-256, backup/restore, prompt-injection blocking, and logging
are untouched. Only the `/chat` endpoint (backend/main.py) and a new
`backend/rag.py` were changed.

## 1. Install Python deps in a fresh venv

The zip's `.venv/` was created on Windows (`Scripts/`, not `bin/`), so it
won't run on macOS — delete it and make a new one:

```bash
cd secure-ai-chatbot
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Make sure Ollama is serving

```bash
ollama serve            # if not already running as a background service
ollama list              # should show qwen3.5:4b and gemma4:31b-cloud
```

## 3. Run the backend

```bash
uvicorn backend.main:app --reload
```

- API: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
- Login with `admin` / `admin123`, use the returned token to authorize
  Swagger, then `/upload` a document and hit `/chat`.

## 4. Frontend

Just open `frontend/index.html` in a browser (or serve it with
`python3 -m http.server 5500` from the `frontend/` folder) — no changes
needed there yet.
