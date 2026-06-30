FROM python:3.11-slim

LABEL org.opencontainers.image.title="Energy-Oracle" \
      org.opencontainers.image.description="Real-time energy consumption prediction API" \
      org.opencontainers.image.version="1.1.0" \
      org.opencontainers.image.authors="devneatharva@gmail.com" \
      org.opencontainers.image.source="https://github.com/atharvadevne123/reflective-lantern"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from app.model import generate_synthetic_data, train_model; df, y = generate_synthetic_data(); train_model(df, y, cv_folds=3)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
