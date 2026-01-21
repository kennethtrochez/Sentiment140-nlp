# Base image
FROM python:3.11-slim

# Working Directory
WORKDIR /app

# Dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn transformers torch protobuf

# Copy application code and model
COPY src ./src
COPY models/distilbert/500k ./models/distilbert/500k

# API port
EXPOSE 8000

# Start API server
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]