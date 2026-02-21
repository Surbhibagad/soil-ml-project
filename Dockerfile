# Base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src
COPY data/ ./data
COPY model/ ./model
# Copy trained model
COPY model.pkl ./model.pkl

# Default command (prediction)
ENTRYPOINT ["python", "src/predict.py"]