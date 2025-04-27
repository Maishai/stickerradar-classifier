FROM python:3.10-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y git

WORKDIR /app
COPY uv.lock /app
COPY pyproject.toml /app
RUN uv sync --locked

COPY serve_model.py /app
COPY models/clip_classifier.pth /app/models/clip_classifier.pth

# Expose port
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8000"]