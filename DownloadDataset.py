from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
import requests
import math

# Configuration
VRT_FILE = "CA_SierraNevada_14_B22.vrt"
OUTPUT_FOLDER = Path("SierraNevada_DEM")
MAX_WORKERS = 8

BATCH_NUMBER = 1  # Change from 1 to 10
TOTAL_BATCHES = 10

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Parse VRT and extract URLs
tree = ET.parse(VRT_FILE)
root = tree.getroot()

urls = set()

for source in root.iter("SourceFilename"):
    if source.text:
        url = source.text.strip()

        if url.startswith("/vsicurl/"):
            url = url[len("/vsicurl/"):]

        if url.startswith(("http://", "https://")):
            urls.add(url)

urls = sorted(urls)

# Select 10% batch
if not 1 <= BATCH_NUMBER <= TOTAL_BATCHES:
    raise ValueError("BATCH_NUMBER must be between 1 and 10")

total_files = len(urls)
batch_start = (BATCH_NUMBER - 1) * total_files // TOTAL_BATCHES
batch_end = BATCH_NUMBER * total_files // TOTAL_BATCHES

batch_urls = urls[batch_start:batch_end]

print(f"Total dataset files: {total_files}")
print(f"Batch: {BATCH_NUMBER}/{TOTAL_BATCHES}")
print(f"Files in batch: {len(batch_urls)}")
print(f"Dataset range: {batch_start + 1} - {batch_end}")


# Download a single tile
def download_tile(url):
    filename = Path(urlparse(url).path).name
    destination = OUTPUT_FOLDER / filename

    # Skip previously downloaded files
    if destination.exists() and destination.stat().st_size > 0:
        return f"Skipped: {filename}"

    temp_file = destination.with_name(filename + ".part")

    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()

            with open(temp_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

        temp_file.replace(destination)
        return f"Downloaded: {filename}"

    except Exception as e:
        temp_file.unlink(missing_ok=True)
        return f"Failed: {filename} ({e})"


# Download selected batch
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [
        executor.submit(download_tile, url)
        for url in batch_urls
    ]

    for future in as_completed(futures):
        print(future.result())

print(f"Batch {BATCH_NUMBER}/{TOTAL_BATCHES} finished.")
