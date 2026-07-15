FROM python:3.11-slim
WORKDIR /app

# Ensure pip is up to date before installing dependencies
RUN pip install --no-cache-dir --upgrade pip

# REMOVED: duckdb, polars
# ADDED: psycopg2-binary for enterprise-grade transactional engine mapping
RUN pip install --no-cache-dir fastapi uvicorn pandas pyarrow psycopg2-binary

COPY fastapi/main.py /app/main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]