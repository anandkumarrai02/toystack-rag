# Use an official Python runtime as a parent image (slim version for smaller size)
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for psycopg2 and other Python packages that require compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . /app

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the .env file to the container (ensure it's in your source directory)
COPY .env /app/.env

# Expose port for Streamlit (default is 8501)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
