"""
Data Collection Script - Text Only (No Audio)
Downloads only text transcripts from datasets
"""

import os
from datasets import load_dataset, Dataset
import json

# Create data directory
DATA_DIR = "raw_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Configuration
MAX_SAMPLES_SADA22 = 5000  

print("=" * 80)
print("Arabic EOU Dataset Collection (Text Only)")
print("=" * 80)
print("\nDownloading 2 datasets:")
print("  1. Saudi Test Samples (all ~1,280)")
print("  2. SADA22 (first 5,000 text transcripts)")
print()

downloaded = []

# Dataset 1: Saudi Dialect Test Samples
print("[1/2] Saudi Dialect Test Samples...")
try:
    dataset = load_dataset(
        "Omartificial-Intelligence-Space/saudi-dialect-test-samples",
        split="train",
        trust_remote_code=True
    )
    print(f"✓ Downloaded {len(dataset)} samples")
    dataset.save_to_disk(os.path.join(DATA_DIR, "saudi_test"))
    downloaded.append(("Saudi Test", len(dataset)))
except Exception as e:
    print(f"✗ Error: {e}")

# Dataset 2: SADA22 - Extract text only, skip audio
print(f"\n[2/2] SADA22 (text transcripts only, first {MAX_SAMPLES_SADA22})...")
try:
    print("  Loading in streaming mode (text only)...")
    
    # Load with streaming, remove audio columns
    dataset_stream = load_dataset(
        "MohamedRashad/SADA22",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    # Collect samples in batches to save memory
    print(f"  Collecting {MAX_SAMPLES_SADA22} samples in batches...")
    BATCH_SIZE = 1000
    all_samples = []
    
    for i, sample in enumerate(dataset_stream):
        if i >= MAX_SAMPLES_SADA22:
            break
        
        # Extract only text fields (skip audio)
        text_sample = {}
        for key, value in sample.items():
            # Skip audio data
            if key not in ['audio', 'sound', 'wav', 'mp3']:
                # Convert to simple types
                if isinstance(value, (str, int, float, bool, type(None))):
                    text_sample[key] = value
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    if isinstance(value[0], (str, int, float)):
                        text_sample[key] = value
                else:
                    text_sample[key] = str(value)
        
        all_samples.append(text_sample)
        
        # Progress update every batch
        if (i + 1) % BATCH_SIZE == 0:
            print(f"    Progress: {i + 1}/{MAX_SAMPLES_SADA22}")
    
    print(f"  Collected {len(all_samples)} samples")
    
    # Convert to dataset
    print("  Converting to dataset format...")
    dataset = Dataset.from_list(all_samples)
    
    print(f"✓ Downloaded {len(dataset)} text samples")
    dataset.save_to_disk(os.path.join(DATA_DIR, "sada22"))
    downloaded.append(("SADA22 (text)", len(dataset)))
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("Download Summary")
print("=" * 80)

if downloaded:
    total = sum(count for _, count in downloaded)
    print(f"\nTotal samples: {total:,}")
    for name, count in downloaded:
        print(f"  - {name}: {count:,}")
    
    # Save info
    info = {
        "total_samples": total,
        "datasets": [{"name": n, "samples": c} for n, c in downloaded],
        "data_type": "text_only",
        "download_date": "2025-12-08"
    }
    
    with open(os.path.join(DATA_DIR, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Info saved to {DATA_DIR}/dataset_info.json")
    print("\n✓ Download Complete!")
    print(f"\nEstimated size: {total:,} samples")
    print("\nNext step: python dataset/data_preprocessing.py")
else:
    print("\n❌ No datasets downloaded successfully")
