import os
import sys
import csv
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage

def download_single_image(rec, output_base_dir, bucket):
    case_id = rec['case_id']
    image_path = rec['image_path']
    label = rec['label']

    target_dir = output_base_dir / label
    target_path = target_dir / f"{case_id}.jpg"

    # Skip if already exists and is non-empty
    if target_path.exists() and target_path.stat().st_size > 0:
        return 'skipped', case_id, image_path, None

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(image_path)
        blob.download_to_filename(str(target_path))
        return 'success', case_id, image_path, None
    except Exception as e:
        return 'failure', case_id, image_path, str(e)

def main():
    base_dir = Path(__file__).resolve().parent
    manifest_files = ['train_manifest.csv', 'val_manifest.csv', 'test_manifest.csv']
    bucket_name = 'dx-scin-public-data'

    records = []
    seen = set()

    for filename in manifest_files:
        filepath = base_dir / filename
        if not filepath.exists():
            print(f"Warning: Manifest file not found: {filepath}", flush=True)
            continue
            
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_id = row['case_id'].strip()
                image_path = row['image_path'].strip()
                label = row['label'].strip()
                
                key = (case_id, image_path, label)
                if key not in seen:
                    seen.add(key)
                    records.append({
                        'case_id': case_id,
                        'image_path': image_path,
                        'label': label
                    })

    total_images = len(records)
    print(f"Loaded {total_images} total image records from manifests.", flush=True)
    if total_images == 0:
        print("No image records found. Exiting.", flush=True)
        return

    print("Initializing Google Cloud Storage anonymous client...", flush=True)
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)

    output_base_dir = base_dir / "images"

    success_count = 0
    failure_count = 0
    skipped_count = 0
    processed_count = 0

    lock = threading.Lock()
    num_workers = 16

    print(f"Starting parallel download with {num_workers} worker threads...", flush=True)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(download_single_image, rec, output_base_dir, bucket)
            for rec in records
        ]

        for future in as_completed(futures):
            status, case_id, image_path, err = future.result()
            with lock:
                processed_count += 1
                if status == 'skipped':
                    skipped_count += 1
                    success_count += 1
                elif status == 'success':
                    success_count += 1
                else:
                    failure_count += 1
                    print(f"Error downloading {image_path} (case_id: {case_id}): {err}", flush=True)

                if processed_count % 100 == 0 or processed_count == total_images:
                    print(
                        f"Processed {processed_count}/{total_images} images "
                        f"(Successes: {success_count}, Failures: {failure_count}, Skipped: {skipped_count})",
                        flush=True
                    )

    print("\n" + "=" * 60, flush=True)
    print("DOWNLOAD COMPLETE", flush=True)
    print(f"Total Processed : {total_images}", flush=True)
    print(f"Successes       : {success_count} (Already existing / Skipped: {skipped_count})", flush=True)
    print(f"Failures        : {failure_count}", flush=True)
    print("=" * 60, flush=True)

if __name__ == '__main__':
    main()
