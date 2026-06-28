"""
Download CAMELS-US dataset from Zenodo.
Source: https://zenodo.org/records/15529996

Downloads essential files for LPU-Stream project:
- Time series forcing data + observed streamflow (Daymet/NLDAS/Maurer)
- Catchment attributes (topo, clim, hydro, soil, vege, geol)
"""

import os
import urllib.request
import sys
import time

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "camels_us")
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://zenodo.org/records/15529996/files"

# Essential files for LPU-Stream
FILES = [
    # Main time series + streamflow data (~12GB, largest file)
    "basin_timeseries_v1p2_metForcing_obsFlow.zip",
    # Catchment attributes (small text files)
    "camels_topo.txt",
    "camels_clim.txt",
    "camels_hydro.txt",
    "camels_soil.txt",
    "camels_vege.txt",
    "camels_geol.txt",
    "camels_name.txt",
    "camels_attributes_v2.0.xlsx",
    "readme.txt",
]

def download_file(filename, dest_dir, max_retries=3):
    """Download a file with progress display and retry logic."""
    url = f"{BASE_URL}/{filename}"
    filepath = os.path.join(dest_dir, filename)

    # Skip if already downloaded
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[SKIP] {filename} already exists ({size_mb:.1f} MB)")
        return True

    print(f"\n[DOWNLOAD] {filename}")
    print(f"  URL: {url}")

    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()

            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(downloaded / total_size * 100, 100)
                    dl_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    elapsed = time.time() - start_time
                    if elapsed > 0 and downloaded > 0:
                        speed = downloaded / elapsed / (1024 * 1024)
                        print(f"\r  Progress: {pct:.1f}% ({dl_mb:.0f}/{total_mb:.0f} MB) "
                              f"| {speed:.1f} MB/s", end="", flush=True)

            urllib.request.urlretrieve(url, filepath, reporthook=report_progress)

            elapsed = time.time() - start_time
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"\n  Done: {size_mb:.1f} MB in {elapsed:.0f}s")
            return True

        except Exception as e:
            print(f"\n  [ERROR] Attempt {attempt}/{max_retries}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            if attempt < max_retries:
                wait = 5 * attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)

    print(f"  [FAILED] Could not download {filename}")
    return False


def main():
    print("=" * 60)
    print("CAMELS-US Dataset Download")
    print("=" * 60)
    print(f"Destination: {DATA_DIR}")
    print(f"Files to download: {len(FILES)}")
    print()

    # Download small attribute files first, then the large zip
    small_files = [f for f in FILES if not f.endswith('.zip')]
    large_files = [f for f in FILES if f.endswith('.zip')]

    results = {}

    # Phase 1: Small files
    print("--- Phase 1: Catchment attributes (small files) ---")
    for f in small_files:
        ok = download_file(f, DATA_DIR)
        results[f] = ok

    # Phase 2: Large zip file
    print("\n--- Phase 2: Time series + streamflow data (large file) ---")
    print("This file is ~12GB and may take a long time to download.")
    for f in large_files:
        ok = download_file(f, DATA_DIR)
        results[f] = ok

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    success = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    for f, ok in results.items():
        status = "OK" if ok else "FAILED"
        size = ""
        fp = os.path.join(DATA_DIR, f)
        if os.path.exists(fp):
            size = f" ({os.path.getsize(fp) / (1024*1024):.1f} MB)"
        print(f"  [{status}] {f}{size}")
    print(f"\nTotal: {success} succeeded, {failed} failed")
    print(f"Data directory: {DATA_DIR}")

    if failed > 0:
        print("\nSome files failed. Re-run the script to retry downloads.")
        sys.exit(1)
    else:
        print("\nAll files downloaded successfully!")
        print("Next step: Extract zip file and run data_preprocessing.py")


if __name__ == "__main__":
    main()
