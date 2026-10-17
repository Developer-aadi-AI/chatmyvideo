# ChatMyVideo

Paste a YouTube link, then ask the video questions. You get short answers with timestamps you can click, a one-click summary, and flashcards for revision.

**Try it:** https://chatmyvideo.vercel.app

I built this because I kept scrubbing back and forth through long lectures and talks looking for the one bit I needed. Now I ask the video instead, and every answer tells me where in the video it came from. Click a timestamp and the player jumps there.

## What it does

- **Ask questions about any captioned YouTube video.** Answers come only from what's said in the video. If the video doesn't cover something, it says so instead of making things up.
- **Clickable timestamps.** Every answer cites moments like `[03:12]`. Clicking one seeks the embedded player and starts playing from there.
- **Summary and Flashcards buttons.** One click gives you a short bullet summary or 8 flashcards in a table.
- **Pick your answer language.** English by default, or Hindi, Hinglish, Urdu, Tamil, Spanish and about 15 more. There's also an option to answer in the video's own language.
- **Follow-up questions work.** The last few messages are sent along, so "what did he say after that?" makes sense.
- **Works with long videos.** For anything over 10 minutes it summarises the video section by section first, so summaries and flashcards cover the whole thing, not only the first few minutes.

It only uses captions the video already has (manual or auto-generated). It never downloads the video or audio, and it can't see what's on screen.

## How it works

1. You paste a link. The backend works out the video ID and fetches the captions through [Supadata](https://supadata.ai).
2. The transcript is split into overlapping chunks of about 850 characters, embedded with `multilingual-e5-small` (through the Hugging Face Inference API), and put into an in-memory FAISS index.
3. When you ask something:
   - **Short videos (≤10 min):** the whole transcript goes to the model.
   - **Long videos:** it searches for the most relevant chunks. For summaries, notes and flashcards it summarises each 5-minute section and then combines those summaries.
4. The answer comes from Groq (`gpt-oss-120b` for answers, `gpt-oss-20b` for background steps). Before it reaches you, every timestamp is checked against the parts of the transcript the model actually saw. Anything that doesn't match is removed.

## Tech stack

| Part | What I used |
|---|---|
| Frontend | Next.js 15, React 19, TypeScript, Tailwind CSS. Hosted on Vercel |
| Backend | Python 3.12, FastAPI, Pydantic. Hosted on Railway |
| LLM | Groq free tier: `openai/gpt-oss-120b` and `openai/gpt-oss-20b` |
| Transcripts | Supadata API (existing captions only) |
| Embeddings | `intfloat/multilingual-e5-small` via Hugging Face |
| Search | FAISS, kept in memory |

## Running it locally

You'll need Python 3.12, Node.js with `pnpm` (or `corepack`), and three free API keys:

- **Groq**: https://console.groq.com/keys
- **Supadata**: https://supadata.ai
- **Hugging Face**: https://huggingface.co/settings/tokens. Make a *fine-grained* token and tick **"Make calls to Inference Providers"**. A plain read token won't work.

### 1. Backend

```bash
cd backend
cp ../.env.example .env      # then fill in your three keys
uv sync
uv run fastapi dev app/main.py
```

The API runs on http://localhost:8000. Check http://localhost:8000/health and it should say `{"status":"ok"}`.

If you don't use `uv`, `pip install -r requirements.txt` and `uvicorn app.main:app --reload` also work.

### 2. Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Open http://localhost:3000. The frontend expects the backend at `http://localhost:8000` unless you set `NEXT_PUBLIC_API_URL` in `frontend/.env.local`.

### Settings

These all live in `backend/.env`. Only the three keys are required.

| Variable | Default | What it's for |
|---|---|---|
| `GROQ_API_KEY` | | Required |
| `SUPADATA_API_KEY` | | Required |
| `HF_TOKEN` | | Required (needs the Inference Providers permission) |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Model for the answers you see |
| `LLM_FAST_MODEL` | `openai/gpt-oss-20b` | Model for section summaries and follow-up rewrites |
| `EMBED_PROVIDER` | `huggingface` | Set to `local` to run the embedding model on your machine (needs ~1.1 GB RAM) |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | Embedding model |
| `RETRIEVER_K` | `4` | How many transcript chunks to search with on long videos |
| `FULL_CONTEXT_CHAR_LIMIT` | `24000` | Short videos over this many characters fall back to search |
| `FRONTEND_ORIGIN` | `http://localhost:3000` | Your frontend URL, for CORS |

Never put any of these keys in the frontend. The only thing the frontend needs is `NEXT_PUBLIC_API_URL`.

## Deploying

I deploy straight from GitHub, so every push to `main` redeploys both parts.

**Backend on Railway**
- Root directory: `backend`
- Start command: `python -m pip install -r requirements.txt && python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Variables: your three keys, plus `FRONTEND_ORIGIN=https://your-app.vercel.app`

**Frontend on Vercel**
- Root directory: `frontend`
- Variable: `NEXT_PUBLIC_API_URL=https://your-backend.up.railway.app` (no trailing slash). Redeploy after changing it.

A note on memory: the default setup uses about 70 MB, because embeddings come from Hugging Face's API. Running the embedding model inside the server (`EMBED_PROVIDER=local`) needs over 1 GB, which crashed my Railway instance on any normal-length video.

## Tests

```bash
cd backend
uv run pytest            # about 175 tests, no API keys needed
uv run ruff check .

cd ../frontend
pnpm lint
pnpm build
```

There are also a few handy scripts in `backend`:

- `uv run python -m app.chat` lets you chat with a video in the terminal.
- `uv run python -m app.transcripts.youtube_transcript "<youtube-url>"` prints a video's transcript.
- `uv run python -m app.evaluate_engine --video talk-en` runs the small eval set in `backend/evals/engine_eval.json`. Fill in the video URLs and expected answers first.

## Limitations

This is a student project and still a work in progress, so a few things to know:

- **Nothing is saved yet.** Loaded videos and cached summaries live in memory and disappear when the backend restarts or redeploys. You just load the video again.
- **The first summary of a long video is slow**, roughly 10–30 seconds, because it summarises every section first. After that it's quick.
- **Groq's free tier is limited** to about 8,000 tokens a minute per model. If lots of people use it at once you may see "The answer service is busy"; wait a few seconds and try again.
- **No captions, no answers.** Videos without any captions show an error.
- **No accounts, library or usage limits yet.** Sign-in with Supabase, a saved library and per-user limits are next on the list.

## Project layout

```
backend/
  app/
    main.py            API routes
    agent/             question answering, language handling, prompts
    ingest/            URL parsing, chunking, search
    transcripts/       Supadata caption fetching
    embeddings/        Hugging Face / local embeddings
    llm/               Groq client
    config.py          settings from .env
  tests/
  evals/
frontend/
  app/                 Next.js pages
  components/          Markdown renderer, timestamp buttons
```
