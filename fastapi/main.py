from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends

# Target environment variables mapped from the updated docker-compose file
PG_HOST = os.getenv("POSTGRES_CACHE_HOST", "operational_cache_postgres")
PG_PORT = int(os.getenv("POSTGRES_CACHE_PORT", "5432"))
PG_DB = os.getenv("POSTGRES_CACHE_DB", "operational_cache")
PG_USER = os.getenv("POSTGRES_CACHE_USER", "cache_user")
PG_PASSWORD = os.getenv("POSTGRES_CACHE_PASSWORD", "cache_password")

# Global placeholder for the active server connection pool
db_container = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the PostgreSQL connection pool.
    Initializes a shared thread-safe pool at startup and tears it down at shutdown.
    """
    try:
        db_container["pool"] = ThreadedConnectionPool(
            minconn=1,
            maxconn=30,
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
        )
        yield
    finally:
        # Cleanly disconnect all pool allocations when uvicorn tears down
        if "pool" in db_container and db_container["pool"]:
            db_container["pool"].closeall()


app = FastAPI(
    title="Taxi Demand Prediction Operational Serving API (PostgreSQL Backend)",
    lifespan=lifespan,
)


def get_db():
    """
    FastAPI dependency injection provider.
    Borrows a connection context from the pool and strictly ensures it returns
    back to the pool even if the underlying route handler triggers an error state.
    """
    pool: ThreadedConnectionPool = db_container.get("pool")
    if pool is None:
        raise HTTPException(
            status_code=500, detail="Database connection pool has not been initialized."
        )

    conn = pool.getconn()
    try:
        yield conn
    finally:
        # Return the socket space back to the shared pool without closing the socket entirely
        pool.putconn(conn)


def clean_record(record: dict) -> dict:
    """Sanitizes database NaN/Timestamp data variants into clean JSON-serializable structures."""
    cleaned = {}
    for key, value in record.items():
        if pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, (pd.Timestamp, datetime)):
            cleaned[key] = value.isoformat()
        else:
            cleaned[key] = value
    return cleaned


@app.get("/health")
def health():
    # Attempt to ping the database to verify active liveness
    try:
        pool: ThreadedConnectionPool = db_container.get("pool")
        if pool:
            conn = pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
            pool.putconn(conn)
            db_alive = True
        else:
            db_alive = False
    except Exception:
        db_alive = False

    return {
        "status": "ok",
        "database_connected": db_alive,
        "database_host": PG_HOST,
    }


@app.get("/predictions/{pickup_location_id}/{prediction_hour_ts}")
def get_prediction(
    pickup_location_id: int,
    prediction_hour_ts: str,
    db=Depends(get_db),
):
    try:
        ts = pd.Timestamp(prediction_hour_ts).floor("h").to_pydatetime()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timestamp string structure: {prediction_hour_ts}",
        )

    query = """
        SELECT * FROM hourly_demand_predictions
        WHERE pickup_location_id = %s AND prediction_hour_ts = %s
    """

    try:
        # RealDictCursor maps column keys directly to dictionary structures natively
        with db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (int(pickup_location_id), ts))
            row = cursor.fetchone()
    except Exception as e:
        # Roll back the transaction state if corrupted so the connection returns clean
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Database operational error: {str(e)}"
        )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No row found for location={pickup_location_id} at window={prediction_hour_ts}",
        )

    return clean_record(dict(row))


@app.get("/predictions/{pickup_location_id}")
def list_predictions(pickup_location_id: int, db=Depends(get_db)):
    query = """
        SELECT * FROM hourly_demand_predictions
        WHERE pickup_location_id = %s
        ORDER BY prediction_hour_ts ASC
    """

    try:
        with db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (int(pickup_location_id),))
            rows = cursor.fetchall()
    except Exception as e:
        # Roll back the transaction state if corrupted so the connection returns clean
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Database operational error: {str(e)}"
        )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No records mapped inside cache for zone location_id={pickup_location_id}",
        )

    records = [clean_record(dict(row)) for row in rows]

    return {
        "pickup_location_id": pickup_location_id,
        "count": len(records),
        "predictions": records,
    }
