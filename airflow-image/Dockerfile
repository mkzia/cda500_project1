FROM apache/airflow:3.2.2-python3.11

ARG ICEBERG_VERSION=1.10.0

USER root

# System-level dependencies configuration
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    curl \
    ca-certificates \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/java \
    && JAVA_HOME_ACTUAL="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")" \
    && ln -s "${JAVA_HOME_ACTUAL}" /opt/java/openjdk

ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Switch back to the non-root execution thread to copy and map python requirements safely
USER airflow

# Copy the requirements file from your host folder context straight into the container workspace layer
COPY requirements.txt /home/airflow/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /home/airflow/requirements.txt

USER root

ENV SPARK_HOME=/home/airflow/.local/lib/python3.11/site-packages/pyspark
ENV ICEBERG_VERSION=${ICEBERG_VERSION}
ENV ICEBERG_SPARK_JAR=${SPARK_HOME}/jars/iceberg-spark-runtime-3.5_2.12-${ICEBERG_VERSION}.jar

# Pull and mount the targeted Iceberg runtime jar bundle extension into the Spark directory structures
RUN test -d "${SPARK_HOME}/jars" \
    && curl -fL \
    "https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/${ICEBERG_VERSION}/iceberg-spark-runtime-3.5_2.12-${ICEBERG_VERSION}.jar" \
    -o "${ICEBERG_SPARK_JAR}" \
    && test -s "${ICEBERG_SPARK_JAR}" \
    && chmod 0644 "${ICEBERG_SPARK_JAR}" \
    && ls -lh "${ICEBERG_SPARK_JAR}"

USER airflow

# Enhanced system-wide environment validation smoke test pass
RUN python -c "import pyspark, yaml, pandas, pyarrow, duckdb, psycopg, IPython, notebook, lightgbm, pyiceberg; print('Airflow + Spark + Iceberg + LightGBM + PyIceberg image OK')"