#!/usr/bin/env python
"""
Upload all 415 images to Roboflow
"""
import requests
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

api_key = "EV7bGjsatm1TZitK9cEG"
dataset_path = Path("C:/Users/luisb/dev/rqd_yolo/data/annotated/dataset_hp_v2")
images = sorted((dataset_path / "images").glob("*"))

workspace = "luis-augusto-silva-bq4bv"
project = "rock-quality-hp"
upload_url = f"https://api.roboflow.com/dataset/{workspace}/{project}/upload"

print("="*70)
print("UPLOADING 415 IMAGES TO ROBOFLOW")
print("="*70)
print(f"\nDataset: {len(images)} images")
print(f"URL: {upload_url}\n")

def upload_image(img_path, index, total):
    """Upload single image"""
    try:
        with open(img_path, 'rb') as f:
            files = {'file': f}
            data = {'api_key': api_key}
            response = requests.post(upload_url, files=files, data=data, timeout=15)
            return {
                'path': img_path,
                'status': response.status_code,
                'success': response.status_code in [200, 201],
                'index': index,
                'total': total
            }
    except Exception as e:
        return {
            'path': img_path,
            'status': 'ERROR',
            'success': False,
            'error': str(e),
            'index': index,
            'total': total
        }

# Upload with 5 concurrent threads
print("Starting upload with 5 parallel connections...\n")
start_time = time.time()
uploaded = 0
failed = 0

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(upload_image, img, i, len(images)) for i, img in enumerate(images)]

    for future in as_completed(futures):
        result = future.result()
        idx = result['index'] + 1

        if result['success']:
            uploaded += 1
            status = "✓"
        else:
            failed += 1
            status = "✗"

        # Progress every 10 images
        if idx % 10 == 0 or idx == 1:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            remaining = (len(images) - idx) / rate if rate > 0 else 0
            print(f"[{idx:3d}/{len(images)}] {status} {result['path'].name:35s} | "
                  f"Speed: {rate:.1f} img/s | ETA: {int(remaining)}s")
        elif status == "✗":
            print(f"[{idx:3d}/{len(images)}] {status} {result['path'].name:35s} | "
                  f"Status: {result['status']}")

# Summary
elapsed = time.time() - start_time
print("\n" + "="*70)
print("UPLOAD COMPLETE!")
print("="*70)
print(f"✓ Uploaded: {uploaded}/{len(images)}")
print(f"✗ Failed: {failed}/{len(images)}")
print(f"⏱ Time: {int(elapsed)}s ({elapsed/60:.1f} minutes)")
print(f"✓ View at: https://app.roboflow.com/{workspace}/{project}")
print("="*70)
