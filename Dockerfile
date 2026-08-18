# ==========================================
# Stage 1️⃣ Builder
# ==========================================
FROM registry.access.redhat.com/hi/python:3.12-builder AS builder
USER root
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .

# Create a virtual environment and install pip dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2️⃣ Final (Rootless & Hardened)
# ==========================================
FROM registry.access.redhat.com/hi/python:3.12 AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    OLLAMA_URL="http://ollama:11434" \
    OLLAMA_MODEL="llama3.2:3b"

USER root
WORKDIR /app

# Copy the venv and set ownership to the rootless user (1001)
COPY --from=builder --chown=1001:0 /opt/venv /opt/venv

# Copy application code
COPY --chown=1001:0 service.py requirements.txt ./

# Switch to the non-root user
USER 1001

EXPOSE 8000

# Execute uvicorn explicitly from the venv binary path
CMD ["/opt/venv/bin/uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
