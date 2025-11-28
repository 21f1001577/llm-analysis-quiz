# Use official Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Selenium + Chrome
RUN apt-get update && apt-get install -y \
    wget unzip curl gnupg \
    chromium-driver chromium \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Start using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
