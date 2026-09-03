# FVG Copilot — Render deployment image for the autonomous runner (v2).
# Builds the Alpaca CLI (Go, static binary) + Python runtime in one image,
# satisfying the required-technology rule (Trading API + Alpaca CLI).
# v2: adds pandas/numpy/pytz/alpaca-py — signal_adapter.py imports live_bot.py,
# which needs them at import time.

FROM golang:1.23-alpine AS cli
RUN CGO_ENABLED=0 go install github.com/alpacahq/cli/cmd/alpaca@latest

FROM python:3.11-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=cli /go/bin/alpaca /usr/local/bin/alpaca

WORKDIR /app
COPY requirements.txt .
# Explicit extras in case requirements.txt lags the branch (groq/pydantic were
# never committed; live_bot.py needs alpaca-py + pandas + numpy + pytz)
RUN pip install --no-cache-dir -r requirements.txt groq pydantic alpaca-py pandas numpy pytz

COPY agents/ ./agents/
COPY fvg_bot.py live_bot.py ./

ENV PYTHONUNBUFFERED=1 \
    STATE_DIR=/data

CMD ["python", "agents/autonomous_runner.py"]
