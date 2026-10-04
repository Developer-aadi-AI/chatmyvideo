# ChatMyVideo

ChatMyVideo turns a YouTube video's transcript into a searchable study companion. This repository is at the initial skeleton stage: the landing page and API health check are implemented; video ingestion and chat are not yet available.

## Local development

### Backend

```powershell
cd backend
uv sync
uv run fastapi dev app/main.py
```

The health endpoint is available at `http://localhost:8000/health`.

### Frontend

```powershell
cd frontend
corepack pnpm install
corepack pnpm dev
```

Open `http://localhost:3000`. The page reports whether it can reach the backend. Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` if the API is not at `http://localhost:8000`.

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

Once both services are deployed, open the Vercel URL and confirm the API status indicator reports that the backend is online. The frontend checks `GET /health`; it does not yet ingest videos.

Do not put secret keys in frontend environment variables. Only variables prefixed with `NEXT_PUBLIC_` should be exposed to the browser.
