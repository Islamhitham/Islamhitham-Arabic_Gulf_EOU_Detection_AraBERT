"""
Data Visualization Script
Visualizes preprocessed Arabic EOU dataset and saves examples
"""

import os
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

# Use a font that supports Arabic
matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "dataset/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("Arabic EOU Dataset Visualization")
print("=" * 80)

# Try to load preprocessed data
DATASET_PATH = "dataset/processed_data/arabic_eou_dataset"

try:
    from datasets import load_from_disk
    dataset = load_from_disk(DATASET_PATH)
    print(f"\n✓ Loaded dataset from {DATASET_PATH}")
    use_hf_format = True
except:
    # Try pickle format
    try:
        with open(f"{DATASET_PATH}.pkl", 'rb') as f:
            dataset = pickle.load(f)
        print(f"\n✓ Loaded dataset from {DATASET_PATH}.pkl")
        use_hf_format = False
    except:
        print(f"\n❌ No preprocessed data found at {DATASET_PATH}")
        print("Please run: python dataset/data_preprocessing.py first")
        exit(1)

# Get data
if use_hf_format:
    train_data = list(dataset['train'])
    val_data = list(dataset['validation'])
    test_data = list(dataset['test'])
else:
    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']

print(f"\nDataset splits:")
print(f"  Train: {len(train_data)}")
print(f"  Validation: {len(val_data)}")
print(f"  Test: {len(test_data)}")

# ============================================================================
# 1. Save Text Examples with EOU Annotations
# ============================================================================
print("\n[1/5] Saving text examples...")

examples_file = os.path.join(OUTPUT_DIR, "data_examples.txt")
with open(examples_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ARABIC EOU DATASET - EXAMPLES\n")
    f.write("=" * 80 + "\n\n")
    
    # Show 20 random examples from training set
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(train_data)), min(20, len(train_data)))
    
    for i, idx in enumerate(sample_indices, 1):
        sample = train_data[idx]
        text = sample['text']
        tokens = sample.get('tokens', text.split())
        labels = sample['labels']
        
        f.write(f"\nExample {i}:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Text: {text}\n")
        f.write(f"Tokens: {len(tokens)}\n")
        f.write(f"EOU markers: {sum(labels)}\n")
        f.write(f"\nToken-by-token breakdown:\n")
        
        for j, (token, label) in enumerate(zip(tokens, labels)):
            eou_marker = " <-- EOU" if label == 1 else ""
            f.write(f"  {j+1:2d}. {token:20s} [{'EOU' if label == 1 else 'O':3s}]{eou_marker}\n")
        
        f.write("\n")

print(f"✓ Saved to: {examples_file}")

# ============================================================================
# 2. Dataset Statistics
# ============================================================================
print("\n[2/5] Generating statistics...")

stats = {
    "total_samples": len(train_data) + len(val_data) + len(test_data),
    "train_samples": len(train_data),
    "val_samples": len(val_data),
    "test_samples": len(test_data)
}

# Calculate token statistics
all_samples = train_data + val_data + test_data
token_counts = [len(s.get('tokens', s['text'].split())) for s in all_samples]
eou_counts = [sum(s['labels']) for s in all_samples]

stats.update({
    "total_tokens": sum(token_counts),
    "total_eou_markers": sum(eou_counts),
    "avg_tokens_per_sample": sum(token_counts) / len(all_samples),
    "avg_eou_per_sample": sum(eou_counts) / len(all_samples),
    "min_tokens": min(token_counts),
    "max_tokens": max(token_counts),
    "eou_ratio": sum(eou_counts) / sum(token_counts)
})

stats_file = os.path.join(OUTPUT_DIR, "dataset_statistics.json")
with open(stats_file, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"✓ Saved to: {stats_file}")

# ============================================================================
# 3. Token Length Distribution Chart
# ============================================================================
print("\n[3/5] Creating token length distribution chart...")

plt.figure(figsize=(10, 6))
plt.hist(token_counts, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Tokens per Sample', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Token Length Distribution', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axvline(stats['avg_tokens_per_sample'], color='red', linestyle='--', 
            label=f'Mean: {stats["avg_tokens_per_sample"]:.1f}')
plt.legend()
plt.tight_layout()

chart1_file = os.path.join(OUTPUT_DIR, "token_length_distribution.png")
plt.savefig(chart1_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved to: {chart1_file}")

# ============================================================================
# 4. EOU per Sample Distribution
# ============================================================================
print("\n[4/5] Creating EOU distribution chart...")

plt.figure(figsize=(10, 6))
eou_counter = Counter(eou_counts)
eou_values = sorted(eou_counter.keys())
eou_frequencies = [eou_counter[v] for v in eou_values]

plt.bar(eou_values, eou_frequencies, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Number of EOU Markers per Sample', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('EOU Markers Distribution', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

chart2_file = os.path.join(OUTPUT_DIR, "eou_distribution.png")
plt.savefig(chart2_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved to: {chart2_file}")

# ============================================================================
# 5. Split Distribution Pie Chart
# ============================================================================
print("\n[5/5] Creating dataset split chart...")

plt.figure(figsize=(8, 8))
sizes = [len(train_data), len(val_data), len(test_data)]
labels = ['Train', 'Validation', 'Test']
colors = ['#ff9999', '#66b3ff', '#99ff99']
explode = (0.05, 0.05, 0.05)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
plt.title('Dataset Split Distribution', fontsize=14, fontweight='bold', pad=20)
plt.axis('equal')
plt.tight_layout()

chart3_file = os.path.join(OUTPUT_DIR, "dataset_splits.png")
plt.savefig(chart3_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved to: {chart3_file}")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION SUMMARY")
print("=" * 80)

summary_file = os.path.join(OUTPUT_DIR, "README.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("ARABIC EOU DATASET VISUALIZATIONS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("FILES GENERATED:\n")
    f.write("-" * 80 + "\n")
    f.write(f"1. data_examples.txt           - 20 annotated examples with EOU labels\n")
    f.write(f"2. dataset_statistics.json     - Complete dataset statistics\n")
    f.write(f"3. token_length_distribution.png - Token count histogram\n")
    f.write(f"4. eou_distribution.png        - EOU markers per sample\n")
    f.write(f"5. dataset_splits.png          - Train/Val/Test split pie chart\n\n")
    
    f.write("DATASET STATISTICS:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total Samples: {stats['total_samples']:,}\n")
    f.write(f"  - Train: {stats['train_samples']:,} ({100*stats['train_samples']/stats['total_samples']:.1f}%)\n")
    f.write(f"  - Validation: {stats['val_samples']:,} ({100*stats['val_samples']/stats['total_samples']:.1f}%)\n")
    f.write(f"  - Test: {stats['test_samples']:,} ({100*stats['test_samples']/stats['total_samples']:.1f}%)\n\n")
    
    f.write(f"Total Tokens: {stats['total_tokens']:,}\n")
    f.write(f"Total EOU Markers: {stats['total_eou_markers']:,}\n")
    f.write(f"Average Tokens per Sample: {stats['avg_tokens_per_sample']:.2f}\n")
    f.write(f"Average EOU per Sample: {stats['avg_eou_per_sample']:.2f}\n")
    f.write(f"Token Range: {stats['min_tokens']} - {stats['max_tokens']}\n")
    f.write(f"EOU Ratio: {100*stats['eou_ratio']:.2f}%\n")

print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}/")
print(f"\nGenerated files:")
print(f"  1. data_examples.txt - Text examples with annotations")
print(f"  2. dataset_statistics.json - Statistics in JSON format")
print(f"  3. token_length_distribution.png - Token length chart")
print(f"  4. eou_distribution.png - EOU markers chart")
print(f"  5. dataset_splits.png - Split distribution chart")
print(f"  6. README.txt - Summary report")

print("\n" + "=" * 80)
print("✓ Visualization Complete!")
print("=" * 80)
print(f"\nView files in: {OUTPUT_DIR}/")
