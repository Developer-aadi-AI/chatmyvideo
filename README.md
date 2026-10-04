# ChatMyVideo

ChatMyVideo turns a YouTube video's transcript into a searchable study companion. The backend can fetch timestamped existing captions through Supadata; end-to-end video ingestion and chat are still in progress.

## Local development

### Configuration and secrets

Copy `.env.example` to `.env` and fill in the required API credentials. The `.env` file is ignored by Git. `backend/app/config.py` loads it once and provides typed settings to the backend; it also uses `backend/.env` as a fallback location for `SUPADATA_API_KEY`. The API keys are required; the optional model and retrieval settings default to the values shown in the config. Run `check_config()` from `app.config` to list any missing required keys without displaying their values.

### Backend

```powershell
cd backend
uv sync
uv run fastapi dev app/main.py
```

The health endpoint is available at `http://localhost:8000/health`.

The `GET /transcript?url=<youtube-url>` endpoint returns the transcript language,
duration in seconds, and timestamped segments with `text`, `start`, and `end`
times in seconds. Supadata is called in native mode, so it only uses existing
captions and does not generate an AI transcript. Successful results and
failures are cached in memory for the lifetime of the backend process so a
video is not requested repeatedly during that run.

Groq chat completions use a process-wide client with a 30-second timeout, two
retries, and temperature `0.2`. The configured `EMBED_MODEL` is loaded once per
process and shared by document and query embedding helpers. E5 helpers apply
`passage: ` and `query: ` prefixes automatically and return normalized vectors.
The default model is `intfloat/multilingual-e5-small`. By default
(`EMBED_PROVIDER=huggingface`) embeddings come from the Hugging Face Inference
API, so the backend needs only ~70 MB of RAM; `HF_TOKEN` must be a token with
the "Make calls to Inference Providers" permission. Set `EMBED_PROVIDER=local`
to run the model in-process instead (needs ~1.1 GB of RAM). Restart the backend after changing the model setting.

Groq completions require the video's transcript language and automatically add
an answer-language instruction based on the latest user question. Stable
script and keyword rules detect supported languages; short or ambiguous
questions fall back to the transcript language, and romanized Hindi is treated
as Hinglish rather than another Latin-script language.

Timestamped transcript segments can be grouped for search with
`app.ingest.chunking.chunk_transcript`. It targets 850 characters with roughly
150 characters of whole-segment overlap. Each chunk has plain text without
timestamps, start/end times in seconds, an ordered `position`, video ID, and
language.

Videos can be indexed once per backend process with `POST /videos/index` and
searched with `GET /videos/{video_id}/search?question=...`. Chunks are embedded
into an in-memory FAISS inner-product index; normalized embeddings make its
scores cosine similarities. Search filters near-duplicate chunks, then returns
the remaining matches in their original video order, including start/end
timestamps. Indexing logs the number of chunks and elapsed time. Restarting the
backend clears these in-memory indexes.

Ask questions about an indexed video with
`POST /videos/{video_id}/ask` and JSON `{"question": "...", "limit": 5}`.
Answers are grounded only in the retrieved timestamped excerpts; transcript
text is treated as untrusted content, and citations are retained only when
their timestamps match an excerpt supplied to the model. The response includes
the answer, cited times, and source excerpts for clickable timestamp UI.
For follow-ups, include a `history` array of recent `{"role": "user"|"assistant",
"content": "..."}` messages in the request. The service uses at most its last
six short messages to rewrite the follow-up for search, but answers the original
wording; it stores no conversation history itself. First questions skip the
rewrite call.

For videos up to 10 minutes, answers use the entire timestamped transcript
rather than only top search results. On longer videos, overview requests such
as summaries, main points, or notes (including supported Hindi/Hinglish
phrasing) are handled by summarizing timestamped sections and synthesizing
those summaries. Section summaries are cached in memory for the backend
process, so later overview questions reuse them. These paths preserve the same
answer-language and citation validation rules as regular Q&A.

The framework-independent engine in `app.engine` exposes `load_video(link)` and
`ask(question, history=None)`. `load_video` returns the video ID, transcript
language, duration, and whether it was newly indexed. `ask` returns the answer,
cited timestamps, and source excerpts. Engine failures are raised as
`EngineError` with a user-friendly message; detailed exceptions are logged.
Questions must contain text and be no longer than 2,000 characters.

Start an interactive terminal chat from the backend directory:

```powershell
uv run python -m app.chat
```

Paste a YouTube link, then ask questions and follow-ups. Type `quit` or `exit`
to end the conversation.

### Engine evaluation

The backend includes a small, editable evaluation set at
`backend/evals/engine_eval.json`: a short English tutorial, long English talk,
Hindi video, and a video expected to have only auto-generated captions, each
with three on-topic questions and one off-topic refusal check. Fill in each
video's `url` and, for every on-topic question, `expected_points`,
`expected_answer`, and approximate `expected_time_seconds` before running it.
The template is intentionally blank so no example link consumes transcript
credits.

From `backend`, run one video at a time (recommended to limit transcript/model
usage):

```powershell
uv run python -m app.evaluate_engine --video tutorial-en
```

Available IDs are `tutorial-en`, `talk-en`, `hindi-video`, and
`auto-captions`. To run the entire configured set, use `--all` explicitly.
The tool reports per-question off-topic refusal, whether a citation is within
20 seconds of the expected time, LLM-judged expected-point coverage, and answer
latency. Each run is appended as one JSON line to the ignored local file
`backend/eval-results/runs.jsonl`; compare these records across runs, or select
a different location with `--results <path>`. Use `--dataset <path>` to run a
copied or revised dataset.

To fetch a transcript from the terminal and print its language, segment count,
duration, and first five timestamped segments:

```powershell
cd backend
uv run python -m app.transcripts.youtube_transcript "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

To confirm the model clients load only once and check question/passage similarity:

```powershell
cd backend
uv run python -m app.check_models
```

### Frontend

```powershell
cd frontend
corepack pnpm install
corepack pnpm dev
```

Open `http://localhost:3000`, paste a YouTube link, and ask questions with
follow-ups. Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` to the backend
base URL if it is not `http://localhost:8000`. For Vercel, configure
`NEXT_PUBLIC_API_URL` in the project's environment variables to the deployed
backend URL (without a trailing slash), and redeploy the frontend. The frontend
does not fall back to localhost in production. Set the backend's
`FRONTEND_ORIGIN` to `https://chatmyvideo.vercel.app` (or your Vercel domain)
to allow browser requests from the deployed app.

## Tests and checks

```powershell
cd backend
uv run pytest
uv run ruff check .
```

```powershell
cd frontend
corepack pnpm lint
corepack pnpm build
```

## Initial deployment

Push this project to a GitHub repository, then create two services connected to that repository:

1. **Railway backend:** set the service root directory to `backend`, use `python -m pip install "fastapi[standard]>=0.115,<1"` as the build command, and use `python -m pip install -r requirements.txt && python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT` as the start command. Set `FRONTEND_ORIGIN` to the deployed Vercel origin (for example, `https://your-app.vercel.app`) and set the service's `PORT` variable to match the target port configured for its Railway domain.
2. **Vercel frontend:** set the project root directory to `frontend`. Set `NEXT_PUBLIC_API_URL` to the Railway public URL, without a trailing slash, and deploy. Redeploy the frontend after changing this variable.

Once both services are deployed, open the Vercel URL and confirm the API status indicator reports that the backend is online. The frontend checks `GET /health`; transcript fetching is currently available through the backend API and terminal command.

Do not put secret keys in frontend environment variables. Only variables prefixed with `NEXT_PUBLIC_` should be exposed to the browser.
