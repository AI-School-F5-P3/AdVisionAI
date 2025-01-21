"""
Analyze class distribution in the dataset, which will be used to finetunnig again
"""

import os
from pathlib import Path

def analyze_split(split_dir: str) -> dict:
    """
    Analyze class distribution in a dataset split.
    
    Args:
        split_dir: Path to the split directory
    
    Returns:
        Dictionary with class counts
    """
    class_counts = {i: 0 for i in range(7)}  # 7 clases (0-6)
    labels_dir = os.path.join(split_dir, 'labels')
    
    if not os.path.exists(labels_dir):
        print(f"Warning: Labels directory not found at {labels_dir}")
        return class_counts
    
    # Analizar solo archivos .txt en el directorio de labels
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, label_file)) as f:
                for line in f:
                    try:
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Invalid line in {label_file}: {line.strip()}")
                        continue
    
    return class_counts

def get_class_names() -> dict:
    """Get class names mapping."""
    return {
        0: 'bershka',
        1: 'corte_ingles',
        2: 'desigual',
        3: 'mango',
        4: 'stradivarius',
        5: 'women_secret',
        6: 'zara'
    }

def print_distribution(split_name: str, counts: dict):
    """Print distribution in a formatted way."""
    class_names = get_class_names()
    total = sum(counts.values())
    
    print(f"\n{split_name} Distribution:")
    print("-" * 50)
    print(f"{'Class':<15} {'Count':>8} {'Percentage':>12}")
    print("-" * 50)
    
    for class_id, count in counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        class_name = class_names[class_id]
        print(f"{class_name:<15} {count:>8} {percentage:>11.2f}%")
    
    print("-" * 50)
    print(f"{'Total':<15} {total:>8}")

def main():
    """Main execution function."""
    # Ruta base del dataset
    base_dir = "data/balanced_dataset_backup_20250120_105413"
    
    # Analizar cada split
    splits = ['train', 'valid', 'test']
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            counts = analyze_split(split_dir)
            print_distribution(split.upper(), counts)
        else:
            print(f"\nWarning: Split directory not found: {split_dir}")

if __name__ == "__main__":
    main()