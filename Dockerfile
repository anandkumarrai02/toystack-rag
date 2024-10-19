# Use an official Python runtime as a parent image
FROM python:3.12.6-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies for building Python packages (for psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev

# Install Python dependencies directly (no virtual environment needed)
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the .env file to the container (ensure it's in your source directory)
COPY .env /app/.env

# Expose port for Streamlit (default port is 8501)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
