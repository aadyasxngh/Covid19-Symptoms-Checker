# Start from a Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install pytest

# Default command (can be overridden in GitHub Actions)
CMD ["pytest"]
