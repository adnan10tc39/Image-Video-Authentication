# Use official slim Python base (Python 3.11)
FROM python:3.11-slim

# metadata
LABEL maintainer="development@myairpbotics.com"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_DIR=/models \
    DEVICE=auto

# system deps commonly required for image/video processing and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    file \
    && rm -rf /var/lib/apt/lists/*

# create app user (non-root) and workdir
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app
COPY --chown=appuser:appuser requirments.txt /app/requirements.txt

# install Python deps
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip --no-cache-dir install -r /app/requirements.txt

# copy application code (do not copy models; mount models at runtime)
COPY --chown=appuser:appuser . /app

# make sure models directory exists and is writable by the appuser
RUN mkdir -p ${MODELS_DIR} && chown -R appuser:appuser ${MODELS_DIR}

# switch to non-root user
USER appuser

# expose port used by uvicorn
EXPOSE 8000

# Healthcheck (runs as root in container runtime, but many orchestrators ignore this user constraint)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
  CMD curl -f http://127.0.0.1:8000/health || exit 1

# default command
# you can override CMD at docker run to change host/port or enable workers
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
