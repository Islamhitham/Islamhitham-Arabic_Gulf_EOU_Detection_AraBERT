"""
Data Preprocessing for EOU Detection
Processes Arabic conversational datasets and creates EOU annotations
"""

import os
import json
import re
from datasets import load_from_disk, Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
import numpy as np

DATA_DIR = "dataset/raw_data"
OUTPUT_DIR = "dataset/processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_arabic_text(text):
    """Normalize Arabic text for consistency"""
    if not isinstance(text, str):
        return ""
    
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    # Normalize alef forms
    text = re.sub(r'[إأآا]', 'ا', text)
    # Normalize yeh
    text = re.sub(r'ى', 'ي', text)
    # Normalize teh marbuta
    text = re.sub(r'ة', 'ه', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_eou_from_punctuation(text):
    """
    Detect end-of-utterance using punctuation marks
    Returns list of token indices that are EOU
    """
    eou_markers = ['.', '?', '!', '؟', '،', '\n']
    tokens = text.split()
    labels = []
    
    for i, token in enumerate(tokens):
        # Check if token ends with EOU punctuation
        is_eou = any(token.endswith(marker) for marker in eou_markers)
        # Also mark last token as EOU
        is_last = (i == len(tokens) - 1)
        labels.append(1 if (is_eou or is_last) else 0)
    
    return tokens, labels

def generate_negative_sample(text_tokens):
    """
    Create a negative sample (incomplete utterance) by truncating the text.
    Returns: (text, tokens, labels) or None if not possible
    """
    # Only truncate if we have enough tokens (at least 3)
    if len(text_tokens) < 3:
        return None
        
    # Cut somewhere in the middle (random split point)
    # Don't pick the last word (that would be the full sentence)
    import random
    split_point = random.randint(1, len(text_tokens) - 1)
    
    truncated_tokens = text_tokens[:split_point]
    
    # Text is just joined tokens (simplified)
    truncated_text = " ".join(truncated_tokens)
    
    # Labels: ALL ZEROS. 
    # The last word of a truncated incomplete sentence is NOT an EOU.
    truncated_labels = [0] * len(truncated_tokens)
    
    return truncated_text, truncated_tokens, truncated_labels

def process_dataset(dataset_path, dataset_name):
    """Process a single dataset"""
    print(f"\nProcessing {dataset_name}...")
    
    try:
        # Try loading as datasets format
        ds = load_from_disk(dataset_path)
        print(f"  Loaded {len(ds)} samples")
    except:
        print(f"  Skipping {dataset_name} - not in expected format")
        return None
    
    processed_samples = []
    
    for idx, sample in enumerate(tqdm(ds, desc=f"  Processing {dataset_name}")):
        # Extract text from different possible field names
        text = None
        for field in ['text', 'transcription', 'transcript', 'sentence', 'utterance']:
            if field in sample and sample[field]:
                text = sample[field]
                break
        
        if not text:
            continue
        
        # Normalize text
        text = normalize_arabic_text(text)
        
        if len(text.strip()) < 5:  # Skip very short texts
            continue
        
        # Create EOU labels
        tokens, labels = detect_eou_from_punctuation(text)
        
        if len(tokens) == 0:
            continue
        
        # 1. Add Positive Sample (Complete)
        processed_samples.append({
            'text': text,
            'tokens': tokens,
            'labels': labels,
            'num_tokens': len(tokens),
            'num_eou': sum(labels),
            'source_dataset': dataset_name,
            'is_augmented': False
        })
        
        # 2. Add Negative Sample (Incomplete) - 50% chance or for every sample?
        # Let's add it for every sample to have a balanced 50/50 dataset of Complete vs Incomplete
        neg = generate_negative_sample(tokens)
        if neg:
            neg_text, neg_tokens, neg_labels = neg
            processed_samples.append({
                'text': neg_text,
                'tokens': neg_tokens,
                'labels': neg_labels,
                'num_tokens': len(neg_tokens),
                'num_eou': sum(neg_labels), # Should be 0
                'source_dataset': dataset_name,
                'is_augmented': True
            })
        
        # Limit samples for faster processing (remove this line for full dataset)
        if len(processed_samples) >= 20000: # Increased limit since we doubled samples
            print(f"  Reached sample limit for {dataset_name}")
            break
    
    print(f"  Processed {len(processed_samples)} samples from {dataset_name}")
    return processed_samples

def create_training_format(samples):
    """Convert to format suitable for training"""
    training_data = []
    
    for sample in samples:
        training_data.append({
            'text': sample['text'],
            'labels': sample['labels'],
            'tokens': sample['tokens']
        })
    
    return training_data

print("=" * 80)
print("Arabic EOU Data Preprocessing")
print("=" * 80)

# Process all available datasets
all_samples = []

datasets_to_process = [
    ("raw_data/saudi_test", "Saudi_Test"),
    ("raw_data/sada22", "SADA22")
]

for dataset_path, dataset_name in datasets_to_process:
    if os.path.exists(dataset_path):
        samples = process_dataset(dataset_path, dataset_name)
        if samples:
            all_samples.extend(samples)
    else:
        print(f"\n⚠ Dataset not found: {dataset_path}")

print(f"\n{'=' * 80}")
print(f"Total samples collected: {len(all_samples)}")

if len(all_samples) == 0:
    print("❌ No samples found! Check if datasets were downloaded correctly.")
    exit(1)

# Calculate statistics
total_tokens = sum(s['num_tokens'] for s in all_samples)
total_eou = sum(s['num_eou'] for s in all_samples)
avg_tokens = total_tokens / len(all_samples)
avg_eou = total_eou / len(all_samples)

print(f"Total tokens: {total_tokens:,}")
print(f"Total EOU markers: {total_eou:,}")
print(f"Average tokens per sample: {avg_tokens:.1f}")
print(f"Average EOU per sample: {avg_eou:.1f}")
print(f"EOU ratio: {100 * total_eou / total_tokens:.2f}%")

# Create train/val/test splits (80/10/10)
np.random.seed(42)
np.random.shuffle(all_samples)

n = len(all_samples)
train_size = int(0.8 * n)
val_size = int(0.1 * n)

train_samples = all_samples[:train_size]
val_samples = all_samples[train_size:train_size + val_size]
test_samples = all_samples[train_size + val_size:]

print(f"\nDataset splits:")
print(f"  Train: {len(train_samples)} samples ({100*len(train_samples)/n:.1f}%)")
print(f"  Validation: {len(val_samples)} samples ({100*len(val_samples)/n:.1f}%)")
print(f"  Test: {len(test_samples)} samples ({100*len(test_samples)/n:.1f}%)")

# Convert to Hugging Face dataset format
train_data = create_training_format(train_samples)
val_data = create_training_format(val_samples)
test_data = create_training_format(test_samples)

# Create dataset dict
dataset_dict = DatasetDict({
    'train': Dataset.from_list(train_data),
    'validation': Dataset.from_list(val_data),
    'test': Dataset.from_list(test_data)
})

# Save processed dataset
output_path = os.path.join(OUTPUT_DIR, "arabic_eou_dataset")
dataset_dict.save_to_disk(output_path)
print(f"\n✓ Saved processed dataset to: {output_path}")

# Save metadata
metadata = {
    "dataset_name": "Arabic EOU Detection Dataset",
    "source_datasets": ["Saudi_Test", "SADA22"],
    "dialect": "Saudi Arabic",
    "task": "End-of-Utterance Detection",
    "total_samples": len(all_samples),
    "splits": {
        "train": len(train_samples),
        "validation": len(val_samples),
        "test": len(test_samples)
    },
    "statistics": {
        "total_tokens": int(total_tokens),
        "total_eou_markers": int(total_eou),
        "avg_tokens_per_sample": float(avg_tokens),
        "avg_eou_per_sample": float(avg_eou),
        "eou_ratio": float(total_eou / total_tokens),
        "augmentation_info": "Includes synthetic incomplete utterances (negative samples)"
    },
    "annotation_method": "heuristic_punctuation_based + synthetic_truncation",
    "preprocessing": [
        "Arabic text normalization",
        "Diacritic removal",
        "Alef/Yeh normalization",
        "EOU detection via punctuation markers",
        "Data augmentation (negative samples creation)"
    ]
}

with open(os.path.join(OUTPUT_DIR, "dataset_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)


print(f"\n{'=' * 80}")
print("✓ Preprocessing complete!")
print("Next step: Run model/train.py to fine-tune the EOU detection model")
