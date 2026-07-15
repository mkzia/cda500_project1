FROM python:3.11-slim

# Set environment variables for stable, non-buffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1

WORKDIR /app

# Install requirements first to leverage Docker cache layers
COPY streamlit/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code
COPY streamlit/app.py /app/app.py

# Correctly formatted JSON array for CMD
CMD ["python", "-X", "faulthandler", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]