import os
import requests


def download_nyc_taxi_data(year, month):
    """Downloads NYC Yellow Taxi Trip Records for a specific year and month."""

    # Format month to be two digits (e.g., 1 -> '01')
    formatted_month = f"{int(month):02d}"
    filename = f"yellow_tripdata_{year}-{formatted_month}.parquet"

    # Define paths
    output_dir = "/opt/airflow/data/incoming"
    file_path = os.path.join(output_dir, filename)

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"Skipping: {filename} already exists at {file_path}")
        return

    # Base URL from TLC
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    url = f"{base_url}{filename}"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading from: {url}")

    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Successfully downloaded and saved to: {file_path}")
        elif response.status_code == 404:
            print(f"Skipped: 404 - Data for {year}-{formatted_month} not available.")
        else:
            print(f"Failed to download. HTTP Status Code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during transmission: {e}")


# Main execution loop
for year in range(2025, 2027):
    for month in range(1, 13):
        # Stop at May 2026 as per your previous requirement
        if year == 2026 and month > 5:
            break
        download_nyc_taxi_data(year, month)
