import os
import time
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import requests
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("watcher_daemon")

BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://airflow-webserver:8080").rstrip("/")
USER_NAME = os.getenv("AIRFLOW_USERNAME")
USER_PASS = os.getenv("AIRFLOW_PASSWORD")

WATCH_DIRECTORY = os.getenv("WATCH_DIRECTORY", "/opt/airflow/data/incoming")
PROCESSED_DIRECTORY = os.path.join(WATCH_DIRECTORY, ".processed")
TARGET_ASSET_URI = os.getenv(
    "TARGET_ASSET_URI",
    "file:///opt/airflow/data/incoming/taxi_data_drop",
)

REQUIRED_KEYS = {
    "dataset",
    "source_filepath",
    "source_filename",
    "source_year",
    "source_month",
    "file_size_bytes",
    "discovered_at_utc",
}

# Added state sets to bound memory and track file thread lock dogpiles
pending_manifests: set[str] = set()
actively_processing: set[str] = set()
cached_asset_id: int | None = None

# Global variables for caching authenticated JWTs
cached_jwt: str | None = None
jwt_expires_at: float = 0.0  # Unix timestamp anchor tracking expiration


def wait_until_file_is_stable(
    path: str, attempts: int = 5, sleep_seconds: float = 0.5
) -> bool:
    """
    Watchdog can fire while a file is still being written.
    This waits until the file size is stable across two checks.
    """
    previous_size = -1

    for _ in range(attempts):
        try:
            current_size = os.path.getsize(path)
        except FileNotFoundError:
            return False

        if current_size > 0 and current_size == previous_size:
            return True

        previous_size = current_size
        time.sleep(sleep_seconds)

    return False


def fetch_authenticated_jwt() -> str | None:
    """
    Implemented JWT local token caching.
    Prevents back-to-back authorization payload thrashing against the Airflow endpoint.
    """
    global cached_jwt, jwt_expires_at

    current_time = time.time()
    # Reuse token if it exists and has more than 30 seconds of headroom remaining
    if cached_jwt and current_time < (jwt_expires_at - 30):
        return cached_jwt

    token_url = f"{BASE_URL}/auth/token"

    try:
        response = requests.post(
            token_url,
            json={"username": USER_NAME, "password": USER_PASS},
            timeout=10,
        )

        if response.status_code in (200, 201):
            token = response.json().get("access_token")

            if token:
                cached_jwt = token
                # Default token longevity safety threshold (assume 15 mins/900s if not specified)
                expires_in = response.json().get("expires_in", 900)
                jwt_expires_at = current_time + expires_in
                return token

            log.error("Auth response did not contain access_token.")
            return None

        log.error("Auth failed: %s %s", response.status_code, response.text)
        return None

    except requests.RequestException as exc:
        log.exception("Connection failure targeting Airflow auth endpoint: %s", exc)
        return None


def find_asset_id_by_uri(headers: dict[str, str], target_uri: str) -> int | None:
    global cached_asset_id

    if cached_asset_id is not None:
        return cached_asset_id

    try:
        response = requests.get(
            f"{BASE_URL}/api/v2/assets", headers=headers, timeout=10
        )

        if response.status_code != 200:
            log.error(
                "Asset registry lookup failed: %s %s",
                response.status_code,
                response.text,
            )
            return None

        assets = response.json().get("assets", [])

        for asset in assets:
            if asset.get("uri") == target_uri:
                cached_asset_id = asset.get("id")
                log.info(
                    "Resolved Airflow asset URI %s to asset_id=%s",
                    target_uri,
                    cached_asset_id,
                )
                return cached_asset_id

        log.warning("Asset URI not found in Airflow registry yet: %s", target_uri)
        return None

    except requests.RequestException as exc:
        log.exception("Registry lookup exception: %s", exc)
        return None


def load_manifest(manifest_path: str) -> dict[str, Any] | None:
    if not wait_until_file_is_stable(manifest_path):
        log.warning("Manifest file is not stable yet; will retry: %s", manifest_path)
        pending_manifests.add(manifest_path)
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        log.error("Malformed manifest JSON in %s: %s", manifest_path, exc)
        return None

    except OSError as exc:
        log.error("Could not read manifest %s: %s", manifest_path, exc)
        pending_manifests.add(manifest_path)
        return None


def validate_manifest(manifest_path: str, manifest_data: dict[str, Any]) -> bool:
    missing = REQUIRED_KEYS - set(manifest_data)

    if missing:
        log.error(
            "Manifest %s missing required properties: %s",
            manifest_path,
            sorted(missing),
        )
        return False

    payload_parquet_path = manifest_data["source_filepath"]

    if not os.path.exists(payload_parquet_path):
        log.error("Declared data file is missing: %s", payload_parquet_path)
        pending_manifests.add(manifest_path)
        return False

    return True


def archive_processed_manifest(manifest_path: str) -> None:
    """
    Promotion helper to safely move files out of the tracking scope
    and completely eliminate bounded internal memory expansion risks.
    """
    try:
        base_name = os.path.basename(manifest_path)
        destination = os.path.join(PROCESSED_DIRECTORY, base_name)
        shutil.move(manifest_path, destination)
        log.debug("Manifest archived to %s", destination)
    except Exception as exc:
        log.error("Failed to archive manifest %s: %s", manifest_path, exc)


def process_manifest(manifest_path: str) -> None:
    manifest_path = str(Path(manifest_path).resolve())

    # Prevent processing dogpiling if an event is already actively evaluating this path
    if manifest_path in actively_processing:
        return

    # Ignore files already moved into the internal hidden processed subdirectory
    if PROCESSED_DIRECTORY in manifest_path:
        return

    actively_processing.add(manifest_path)

    try:
        manifest_data = load_manifest(manifest_path)

        if manifest_data is None:
            return

        if not validate_manifest(manifest_path, manifest_data):
            return

        access_token = fetch_authenticated_jwt()

        if not access_token:
            pending_manifests.add(manifest_path)
            return

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        asset_id = find_asset_id_by_uri(headers, TARGET_ASSET_URI)

        if not asset_id:
            pending_manifests.add(manifest_path)
            return

        event_payload = {
            "asset_id": asset_id,
            "extra": manifest_data,
        }

        try:
            response = requests.post(
                f"{BASE_URL}/api/v2/assets/events",
                json=event_payload,
                headers=headers,
                timeout=10,
            )

            if response.status_code in (200, 201):
                pending_manifests.discard(manifest_path)
                log.info(
                    "Airflow asset event emitted for manifest: %s",
                    os.path.basename(manifest_path),
                )
                # Promote/archive the file cleanly rather than keeping string tracks in memory
                archive_processed_manifest(manifest_path)
                return

            log.error(
                "Asset event emission rejected: %s %s",
                response.status_code,
                response.text,
            )
            pending_manifests.add(manifest_path)

        except requests.RequestException as exc:
            log.exception("Airflow event API transport failure: %s", exc)
            pending_manifests.add(manifest_path)

    finally:
        # Guarantee removal from active locks block list regardless of failure state
        actively_processing.discard(manifest_path)


class InboundManifestStagingHandler(FileSystemEventHandler):
    # Handler utilizes safety checkpoints to avoid multi-event execution congestion
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".manifest.json"):
            process_manifest(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".manifest.json"):
            process_manifest(event.src_path)


def process_existing_manifests() -> None:
    for root, _, files in os.walk(WATCH_DIRECTORY):
        # Skip the hidden promotional directory entirely when performing reconciliation walks
        if PROCESSED_DIRECTORY in root:
            continue
        for filename in files:
            if filename.endswith(".manifest.json"):
                process_manifest(os.path.join(root, filename))


def retry_pending_manifests() -> None:
    for pending_path in list(pending_manifests):
        if os.path.exists(pending_path):
            process_manifest(pending_path)
        else:
            pending_manifests.discard(pending_path)


if __name__ == "__main__":
    os.makedirs(WATCH_DIRECTORY, exist_ok=True)
    os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)

    log.info("Starting mock Lambda watcher daemon")
    log.info("Watching directory: %s", WATCH_DIRECTORY)
    log.info("Target Airflow base URL: %s", BASE_URL)
    log.info("Target Airflow asset URI: %s", TARGET_ASSET_URI)

    process_existing_manifests()

    handler = InboundManifestStagingHandler()
    observer = Observer()
    observer.schedule(handler, WATCH_DIRECTORY, recursive=True)
    observer.start()

    try:
        while True:
            retry_pending_manifests()
            time.sleep(10)

    except KeyboardInterrupt:
        log.info("Stopping watcher daemon")

    finally:
        observer.stop()
        observer.join()
