FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    DEVICE=cpu \
    MODELS_DIR=models \
    HF_MODEL_NAMESPACE=hansaprem703

WORKDIR /app

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application, inference pipeline, and model metadata
COPY backend/ backend/
COPY src/ src/
COPY data/ data/
COPY models/ models/

EXPOSE 8080

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
