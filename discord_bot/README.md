# DocIntel Discord Bot

Thin Discord client for the DocIntel FastAPI backend. The bot forwards user messages to `/routed_query` and posts the backend response back to Discord.

## Behavior

- Uses the Discord channel ID as the `session_id` for stable per-channel conversations.
- Holds no local state; all memory is persisted by the backend.

## Requirements

- Python 3.10+
- A Discord bot token with the Message Content intent enabled
- Running DocIntel backend (local or Docker)

## Run with Docker Compose

From the repository root:
```bash
docker compose up --build
```

The bot will talk to `http://backend:8000` inside the Docker network.

## Run locally

1. Create and populate `.env`:
```bash
cp .env.example .env
```

2. Install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the bot:
```bash
python bot.py
```

If the backend is local, set `DOCINTEL_API_BASE=http://localhost:8000`.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| DISCORD_BOT_TOKEN | yes | - | Discord bot token |
| DOCINTEL_API_BASE | no | http://localhost:8000 | Base URL for DocIntel API |
| DOCINTEL_REQUEST_TIMEOUT | no | 15 | Timeout (seconds) for API calls |

## Notes

- Keep the bot running in a separate process or container from the backend.
- Avoid logging sensitive user content; persistence is handled by the backend.
