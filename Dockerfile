# FastAPI image for a container host (Render, Fly.io, Railway, Cloud Run, ...).
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY dfs ./dfs
RUN pip install --no-cache-dir .
# Configuration comes from environment variables: DATABASE_URL_READONLY, ALLOWED_ORIGINS.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn dfs.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
