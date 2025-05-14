FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy the GeoClimate Fetcher package and install it
COPY . .
RUN pip install -e .

# Expose the Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 