# Use Python 3.8 slim image as base
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DOCKER_API_VERSION=1.41

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY main.py .

# Create necessary directories for logging
RUN mkdir -p /var/log/effiscale

# Create a non-root user
RUN useradd -m effiscale && \
    chown -R effiscale:effiscale /app /var/log/effiscale

# Switch to non-root user
USER effiscale

# Expose ports for services
EXPOSE 8001 8002 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Set entrypoint
ENTRYPOINT ["python"]

# Set default command
CMD ["main.py"]