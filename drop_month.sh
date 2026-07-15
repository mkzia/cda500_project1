#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: ./drop_month.sh YEAR MONTH"
    echo "Example: ./drop_month.sh 2026 4"
    exit 1
fi

YEAR=$1
RAW_MONTH=$2

if ! [[ "$YEAR" =~ ^[0-9]{4}$ ]]; then
    echo "❌ Error: YEAR must be a valid 4-digit number, e.g. 2026."
    exit 1
fi

if ! [[ "$RAW_MONTH" =~ ^[0-9]{1,2}$ ]]; then
    echo "❌ Error: MONTH must evaluate as an integer from 1 to 12."
    exit 1
fi

MONTH_NUM=$((10#$RAW_MONTH))

if [ "$MONTH_NUM" -lt 1 ] || [ "$MONTH_NUM" -gt 12 ]; then
    echo "❌ Error: MONTH must evaluate within standard 1 to 12 ranges."
    exit 1
fi

MONTH=$(printf "%02d" "$MONTH_NUM")
TARGET_DIR="./data/incoming"
PARQUET_FILE="yellow_tripdata_${YEAR}-${MONTH}.parquet"

if [ ! -f "${TARGET_DIR}/${PARQUET_FILE}" ]; then
    echo "❌ Error: Place ${PARQUET_FILE} inside ${TARGET_DIR} first."
    exit 1
fi

if stat -c%s "${TARGET_DIR}/${PARQUET_FILE}" >/dev/null 2>&1; then
    FILE_SIZE=$(stat -c%s "${TARGET_DIR}/${PARQUET_FILE}")
else
    FILE_SIZE=$(stat -f%z "${TARGET_DIR}/${PARQUET_FILE}")
fi

cat > "${TARGET_DIR}/yellow_tripdata_${YEAR}_${MONTH}.manifest.json" <<EOF
{
  "dataset": "yellow_taxi",
  "source_filepath": "/opt/airflow/data/incoming/${PARQUET_FILE}",
  "source_filename": "${PARQUET_FILE}",
  "source_year": ${YEAR},
  "source_month": ${MONTH_NUM},
  "file_size_bytes": ${FILE_SIZE},
  "discovered_at_utc": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo "📡 Handshake manifest generated for ${PARQUET_FILE}."
