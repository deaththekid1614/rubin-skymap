FROM python:3.11-slim

# Install system deps needed by some packages (e.g. lightgbm, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories that must exist at runtime
RUN mkdir -p models data/raw data/processed mlruns

EXPOSE 8000

CMD ["uvicorn", "rubin_skymap.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
