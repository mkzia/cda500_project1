#!/bin/bash

# Configuration: Adjust these dates as needed (format YYYY-MM-DD)
START_DATE="2026-07-01"
END_DATE="2026-07-13"
DAG_ID="4_ml_batch_demand_score"

echo "🚀 Triggering $DAG_ID from $START_DATE to $END_DATE..."

# Convert dates to seconds since epoch for iteration
start_ts=$(date -j -f "%Y-%m-%d" "$START_DATE" +%s)
end_ts=$(date -j -f "%Y-%m-%d" "$END_DATE" +%s)

# Loop through each day
curr_ts=$start_ts
while [ $curr_ts -le $end_ts ]; do

    # Format the current date
    day_str=$(date -j -f "%s" "$curr_ts" +%Y-%m-%d)

    # Loop through each hour (0 to 23)
    for hour in {0..23}; do
        # Format hour to two digits (e.g., 00, 01, ..., 23)
        formatted_hour=$(printf "%02d" $hour)
        target_ts="${day_str}T${formatted_hour}:00:00"

        echo "📅 Triggering: $target_ts"

        docker compose exec airflow-webserver airflow dags trigger \
            "$DAG_ID" \
            --conf "{\"real_now\": \"$target_ts\", \"months_back\": 2}"
    done

    # Increment by 1 day (86400 seconds)
    curr_ts=$((curr_ts + 86400))
done

echo "✅ All triggers submitted for the range $START_DATE to $END_DATE."