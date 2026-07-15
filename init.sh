#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Local teaching environment initializer
# Purpose:
# - Create a simple project layout
# - Create a simple Airflow auth password file
# - Create a local config file for the data engineering workflow
# - Create a .env file used by docker-compose.yml
#
# This is intentionally simple for teaching.
# Do not treat these credentials as production secrets.
# -----------------------------------------------------------------------------

AIRFLOW_ADMIN_USERNAME="admin"
AIRFLOW_ADMIN_PASSWORD="admin"

POSTGRES_USER="airflow_user"
POSTGRES_PASSWORD="airflow_pass"
POSTGRES_DB="airflow_metadata"

AIRFLOW_IMAGE_NAME="airflow-spark-iceberg-teaching:3.2.2"

# On Linux, using the host UID avoids root-owned files in mounted folders.
# On macOS/Windows Docker Desktop this is usually harmless.
AIRFLOW_UID="$(id -u)"

# Stable development keys for repeatable classroom behavior.
# Replace these for any non-local/non-teaching use.
AIRFLOW_FERNET_KEY="81HqDtbqAywKSOumSha3BhWNOdQ26slT6K0YaZeZyPs="
AIRFLOW_API_SECRET_KEY="dev_api_secret_key_change_me_for_nonlocal_use_123456"
AIRFLOW_JWT_SECRET="dev_jwt_secret_change_me_for_nonlocal_use_1234567890"
JUPYTER_TOKEN="dev_token"

echo "Creating local teaching project folders..."

mkdir -p \
  ./airflow-image \
  ./dags \
  ./jobs \
  ./sql \
  ./config \
  ./data/incoming/taxi_data_drop \
  ./data/warehouse \
  ./logs \
  ./airflow_auth \
  ./notebooks

echo "Creating Simple Auth Manager password file..."

cat > ./airflow_auth/simple_auth_manager_passwords.json.generated <<EOF
{"${AIRFLOW_ADMIN_USERNAME}": "${AIRFLOW_ADMIN_PASSWORD}"}
EOF

chmod 600 ./airflow_auth/simple_auth_manager_passwords.json.generated

echo "Creating local data engineering configuration..."

cat > ./config/local.yml <<'EOF'
environment: local

catalog:
  spark_catalog_name: nyc
  logical_name: nyc_catalog
  type: hadoop
  warehouse_path: /opt/airflow/data/warehouse

storage:
  incoming_directory: /opt/airflow/data/incoming
  warehouse_directory: /opt/airflow/data/warehouse

assets:
  incoming_taxi_data_uri: file:///opt/airflow/data/incoming/taxi_data_drop
  bronze_taxi_data_uri: iceberg://bronze.yellow_taxi_raw
  silver_taxi_data_uri: iceberg://silver.yellow_taxi_trips
EOF

echo "Creating docker-compose .env file..."

cat > ./.env <<EOF
AIRFLOW_UID=${AIRFLOW_UID}
AIRFLOW_IMAGE_NAME=${AIRFLOW_IMAGE_NAME}

POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=${POSTGRES_DB}

AIRFLOW_ADMIN_USERNAME=${AIRFLOW_ADMIN_USERNAME}
AIRFLOW_ADMIN_PASSWORD=${AIRFLOW_ADMIN_PASSWORD}

AIRFLOW_FERNET_KEY=${AIRFLOW_FERNET_KEY}
AIRFLOW_API_SECRET_KEY=${AIRFLOW_API_SECRET_KEY}
AIRFLOW_JWT_SECRET=${AIRFLOW_JWT_SECRET}

JUPYTER_TOKEN=${JUPYTER_TOKEN}
EOF

chmod 600 ./.env

# Keep most teaching folders easy to edit from the host.
chmod -R u+rwX,g+rwX \
  ./dags \
  ./jobs \
  ./sql \
  ./config \
  ./data \
  ./logs \
  ./notebooks \
  ./airflow-image

echo
echo "Local curriculum lakehouse structure created successfully."
echo
echo "Airflow login:"
echo "  username: ${AIRFLOW_ADMIN_USERNAME}"
echo "  password: ${AIRFLOW_ADMIN_PASSWORD}"
echo
echo "Jupyter token:"
echo "  ${JUPYTER_TOKEN}"
echo
echo "Next steps:"
echo "  1. Put the Dockerfile in ./airflow-image/Dockerfile"
echo "  2. Put docker-compose.yml in the project root"
echo "  3. Run: docker compose build"
echo "  4. Run: docker compose up"