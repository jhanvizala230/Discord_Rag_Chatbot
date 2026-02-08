FROM python:3.11

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY discord_bot/requirements.txt ./discord_bot/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r ./discord_bot/requirements.txt

COPY discord_bot/ ./discord_bot/

WORKDIR /app/discord_bot

CMD ["python", "bot.py"]
