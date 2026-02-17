# Use official Python image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY app/ ./app/
COPY models/ ./models/

# Expose port for the API
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the app 
# For FastAPI with Uvicorn:
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# For Flask, :
# CMD ["python", "app/main.py"]
