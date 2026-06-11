FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build deps first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install ".[dashboard]"

COPY dashboard ./dashboard
COPY demo ./demo

RUN mkdir -p /data

EXPOSE 8000 8501

CMD ["uvicorn", "sentinel.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
