"""
Quick Data Preview
Shows what the downloaded datasets look like before preprocessing
"""

import os
import json

print("=" * 80)
print("Quick Dataset Preview")
print("=" * 80)

# Check for downloaded data
DATA_DIR = "dataset/raw_data"

if not os.path.exists(DATA_DIR):
    DATA_DIR = "raw_data"  # Alternative location

if not os.path.exists(DATA_DIR):
    print("\n⚠ No data downloaded yet.")
    print("Download is in progress. Once complete, run:")
    print("  python dataset/visualize_data.py")
    exit(0)

print(f"\nChecking {DATA_DIR}...")

# Try to load datasets
found_any = False

# Check Saudi Test Samples
saudi_test_path = os.path.join(DATA_DIR, "saudi_test")
if os.path.exists(saudi_test_path):
    print(f"\n✓ Found: Saudi Test Samples")
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(saudi_test_path)
        print(f"  Samples: {len(dataset)}")
        
        # Show first 5 examples
        print("\n  Sample Data:")
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            text = sample.get('text', sample.get('sentence', str(sample)[:100]))
            print(f"    {i+1}. {text}")
        
        found_any = True
    except Exception as e:
        print(f"  Error loading: {e}")

# Check SADA22
sada22_path = os.path.join(DATA_DIR, "sada22")
if os.path.exists(sada22_path):
    print(f"\n✓ Found: SADA22")
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(sada22_path)
        print(f"  Samples: {len(dataset)}")
        
        # Show first 3 examples
        print("\n  Sample Data:")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            text = sample.get('text', sample.get('transcript', str(sample)[:100]))
            print(f"    {i+1}. {text}")
        
        found_any = True
    except Exception as e:
        print(f"  Error loading: {e}")

# Check Free Dialogue
free_dialogue_path = os.path.join(DATA_DIR, "free_dialogue")
if os.path.exists(free_dialogue_path):
    print(f"\n✓ Found: Free Dialogue")
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(free_dialogue_path)
        print(f"  Samples: {len(dataset)}")
        
        # Show first 3 examples
        print("\n  Sample Data:")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            text = sample.get('text', sample.get('transcription', str(sample)[:100]))
            print(f"    {i+1}. {text}")
        
        found_any = True
    except Exception as e:
        print(f"  Error loading: {e}")

if not found_any:
    print("\n⚠ No datasets found yet.")
    print("Download may still be in progress.")
else:
    print("\n" + "=" * 80)
    print("✓ Preview Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Wait for download to complete")
    print("2. Run preprocessing: python dataset/data_preprocessing.py")
    print("3. Run full visualization: python dataset/visualize_data.py")
