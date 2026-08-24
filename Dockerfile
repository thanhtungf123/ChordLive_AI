FROM python:3.10-slim

WORKDIR /app

# Install ffmpeg for yt-dlp
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run FastAPI with Uvicorn using the PORT env var (Render default is usually 10000)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
