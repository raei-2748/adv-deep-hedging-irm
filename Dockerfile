# Dockerfile for deep hedging project with deterministic dependencies
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install pip-tools
RUN python -m pip install --no-cache-dir pip-tools

# Copy requirements files
COPY requirements.in requirements.lock ./

# Install dependencies from lockfile for deterministic builds
RUN pip-sync requirements.lock

# Copy project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "experiment.py"]
