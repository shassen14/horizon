# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# 1. System Dependencies (GCC/Make needed for TA-Lib)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Install TA-Lib C Library (Source Build for ARM/Pi compatibility)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 3. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 4. Copy Dependency Definitions
COPY requirements.in .
COPY packages/ ./packages/

# 5. Install Dependencies & Local Packages
# We assume requirements.in includes 'TA-Lib' (the wrapper)
RUN uv pip install --system -r requirements.in
RUN uv pip install --system -e ./packages/database
RUN uv pip install --system -e ./packages/quant_lib
RUN uv pip install --system -e ./packages/ml_core

# 6. Copy Application Code
COPY apps/ ./apps/

# Default command (will be overridden by docker-compose)
CMD ["python", "apps/api_server/main.py"]