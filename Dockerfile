FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/synthetic models temp logs

# Expose the API port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Run the API server
CMD ["python", "-m", "procurement_granite.api.app"] 