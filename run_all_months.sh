#!/bin/bash
set -euo pipefail

# Define the start and end years
START_YEAR=2025
END_YEAR=2026

# Loop through years
for year in $(seq $START_YEAR $END_YEAR); do
    # Determine the range of months for the current year
    if [ "$year" -eq 2025 ]; then
        start_month=1
        end_month=12
    else
        # Only go up to May (5) for 2026
        start_month=1
        end_month=5
    fi

    # Loop through months
    for month in $(seq $start_month $end_month); do
        echo ">>> Triggering drop_month.sh for $year-$month"
        ./drop_month.sh "$year" "$month"
        sleep 40
    done
done

echo "✅ All months processed."