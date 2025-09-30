# Use lightweight Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for PyTorch, OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Copy requirement file and install deps
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command: run FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
