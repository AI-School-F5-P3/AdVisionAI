"""
Verify and clean augmented dataset.
"""

import os
from pathlib import Path
import logging
from datetime import datetime
import shutil

class DatasetVerifier:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def analyze_distribution(self):
        """Analyze class distribution in the dataset."""
        stats = {split: {i: 0 for i in range(7)} for split in ['train', 'valid', 'test']}
        total_files = {split: 0 for split in ['train', 'valid', 'test']}
        
        for split in ['train', 'valid', 'test']:
            labels_dir = self.dataset_path / split / 'labels'
            if not labels_dir.exists():
                continue
                
            for label_file in labels_dir.glob('*.txt'):
                total_files[split] += 1
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            class_id = int(line.split()[0])
                            stats[split][class_id] += 1
                except Exception as e:
                    self.logger.warning(f"Error processing {label_file}: {e}")

        # Print statistics
        self.logger.info("\nDataset Distribution:")
        class_names = {
            0: 'bershka',
            1: 'corte_ingles',
            2: 'desigual',
            3: 'mango',
            4: 'stradivarius',
            5: 'women_secret',
            6: 'zara'
        }
        
        for split in ['train', 'valid', 'test']:
            total = sum(stats[split].values())
            self.logger.info(f"\n{split.upper()} Split (Total files: {total_files[split]}):")
            self.logger.info("-" * 50)
            self.logger.info(f"{'Class':<15} {'Count':>8} {'Percentage':>12}")
            self.logger.info("-" * 50)
            
            for class_id, count in stats[split].items():
                percentage = (count / total * 100) if total > 0 else 0
                self.logger.info(f"{class_names[class_id]:<15} {count:>8} {percentage:>11.1f}%")
            
        return stats, total_files

    def clean_failed_augmentations(self):
        """Remove failed augmentation attempts."""
        cleaned = 0
        train_dir = self.dataset_path / 'train'
        
        # Verificar pares imagen-etiqueta
        images = set(f.stem for f in (train_dir / 'images').glob('*'))
        labels = set(f.stem for f in (train_dir / 'labels').glob('*'))
        
        # Encontrar archivos sin pareja
        orphaned_images = images - labels
        orphaned_labels = labels - images
        
        # Eliminar archivos sin pareja
        for stem in orphaned_images:
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = train_dir / 'images' / f"{stem}{ext}"
                if img_path.exists():
                    img_path.unlink()
                    cleaned += 1
                    self.logger.info(f"Removed orphaned image: {img_path}")
                    
        for stem in orphaned_labels:
            label_path = train_dir / 'labels' / f"{stem}.txt"
            if label_path.exists():
                label_path.unlink()
                cleaned += 1
                self.logger.info(f"Removed orphaned label: {label_path}")
        
        self.logger.info(f"\nCleaned {cleaned} orphaned files")
        return cleaned

def main():
    try:
        # Encontrar el último dataset aumentado
        data_dir = Path("data")
        augmented_datasets = sorted(
            [d for d in data_dir.glob("augmented_dataset_*") if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if not augmented_datasets:
            print("No augmented datasets found")
            return
            
        latest_dataset = augmented_datasets[0]
        print(f"\nProcessing latest dataset: {latest_dataset}")
        
        # Analizar y limpiar
        verifier = DatasetVerifier(latest_dataset)
        
        print("\nDistribución inicial:")
        initial_stats, initial_files = verifier.analyze_distribution()
        
        print("\nLimpiando archivos fallidos...")
        cleaned = verifier.clean_failed_augmentations()
        
        if cleaned > 0:
            print("\nDistribución después de limpieza:")
            final_stats, final_files = verifier.analyze_distribution()
            
        print("\nProceso completado!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()