"""
Dataset augmentation tool for logo detection.

This module provides advanced augmentation techniques specifically designed
for logo detection tasks, with emphasis on maintaining logo integrity
while increasing dataset diversity.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from PIL import Image
import shutil
from tqdm import tqdm
import logging
from datetime import datetime
from typing import Dict, List, Tuple

class LogoAugmentor:
    def __init__(self, base_path: str = "data/balanced_dataset_backup_20250120_105413"):
        """Initialize Logo Augmentor with dataset path."""
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {base_path}")
            
        self.output_path = self.base_path.parent / f"augmented_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.setup_logging()
        
        self.logger.info(f"Initializing augmentor with dataset: {self.base_path}")
        self.logger.info(f"Output will be saved to: {self.output_path}")
        
        # Check dataset structure
        required_dirs = [
            self.base_path / 'train' / 'images',
            self.base_path / 'train' / 'labels',
            self.base_path / 'valid' / 'images',
            self.base_path / 'valid' / 'labels',
            self.base_path / 'test' / 'images',
            self.base_path / 'test' / 'labels'
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
        
        # Target counts for balancing via augmentation
        self.target_counts = {
            0: 250,  # bershka
            1: 250,  # corte_ingles
            2: 250,  # desigual
            3: 250,  # mango
            4: 250,  # stradivarius
            5: 250,  # women_secret
            6: 250,  # zara
        }

    def setup_logging(self):
        """Configure logging system."""
        log_dir = Path("logs/augmentation")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"augmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_augmentation_pipeline(self, class_id: int) -> A.Compose:
        """
        Get augmentation pipeline tailored for each class.
        More aggressive augmentation for minority classes.
        """
        # Base transformations for all classes
        base_transform = [
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        ]

        # Specific augmentations for minority classes (women_secret, desigual)
        if class_id in [5, 2]:  # women_secret, desigual
            transform = base_transform + [
                A.Affine(
                    scale=(0.8, 1.2),
                    rotate=(-30, 30),
                    translate_percent=(-0.1, 0.1),
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.7
                ),
                A.RandomShadow(p=0.3),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            ]
        else:
            transform = base_transform + [
                A.Affine(
                    scale=(0.85, 1.15),
                    rotate=(-15, 15),
                    translate_percent=(-0.1, 0.1),
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
            ]

        return A.Compose(transform, bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))

    def process_bbox_file(self, bbox_path: Path) -> List[Tuple[int, List[float]]]:
        """Process YOLO format bbox file."""
        bboxes = []
        try:
            with open(bbox_path, 'r') as f:
                content = f.read().strip()
                self.logger.debug(f"Content of {bbox_path}: {content}")
                if not content:  # Skip empty files
                    return bboxes
                    
                for line in content.split('\n'):
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:]]
                        bboxes.append((class_id, bbox))
                    else:
                        self.logger.warning(f"Invalid line in {bbox_path}: {line}")
                        
            self.logger.debug(f"Processed bboxes from {bbox_path}: {bboxes}")
            return bboxes
            
        except Exception as e:
            self.logger.error(f"Error processing {bbox_path}: {e}")
            return bboxes

    def augment_sample(self, img_path: Path, bbox_path: Path, class_id: int, 
                      transform: A.Compose, output_idx: int):
        """Apply augmentation to a single sample."""
        try:
            # Read image and bboxes
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = self.process_bbox_file(bbox_path)
            
            # Prepare data for albumentations
            boxes = [bbox for _, bbox in bboxes]
            class_labels = [cid for cid, _ in bboxes]
            
            # Apply augmentation
            transformed = transform(
                image=image,
                bboxes=boxes,
                class_labels=class_labels
            )
            
            # Save augmented image
            aug_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
            aug_img_path = self.output_path / 'train/images' / f'{img_path.stem}_aug_{output_idx}.jpg'
            cv2.imwrite(str(aug_img_path), aug_image)
            
            # Save augmented bboxes
            aug_bbox_path = self.output_path / 'train/labels' / f'{img_path.stem}_aug_{output_idx}.txt'
            with open(aug_bbox_path, 'w') as f:
                for bbox, class_label in zip(transformed['bboxes'], transformed['class_labels']):
                    bbox_str = ' '.join([str(x) for x in bbox])
                    f.write(f'{class_label} {bbox_str}\n')
                    
        except Exception as e:
            self.logger.error(f"Error augmenting {img_path}: {e}")

    def augment_dataset(self):
        """Augment the dataset to balance classes."""
        try:
            # Create output directory structure
            for split in ['train', 'valid', 'test']:
                for subdir in ['images', 'labels']:
                    (self.output_path / split / subdir).mkdir(parents=True, exist_ok=True)

            # Copy original dataset
            self.logger.info("Copying original dataset...")
            for split in ['train', 'valid', 'test']:
                for subdir in ['images', 'labels']:
                    src_dir = self.base_path / split / subdir
                    dst_dir = self.output_path / split / subdir
                    for file in src_dir.glob('*'):
                        shutil.copy2(file, dst_dir)

            # Count existing samples per class in train split
            class_counts = {i: 0 for i in range(7)}
            train_labels_dir = self.base_path / 'train/labels'
            for label_file in train_labels_dir.glob('*.txt'):
                bboxes = self.process_bbox_file(label_file)
                for class_id, _ in bboxes:
                    class_counts[class_id] += 1

            # Calculate needed augmentations
            aug_needed = {
                class_id: max(0, self.target_counts[class_id] - count)
                for class_id, count in class_counts.items()
            }

            self.logger.info(f"Original class distribution: {class_counts}")
            self.logger.info(f"Augmentations needed: {aug_needed}")

            # Perform augmentation for each class
            for class_id, num_aug in aug_needed.items():
                if num_aug <= 0:
                    continue

                self.logger.info(f"Augmenting class {class_id}")
                transform = self.get_augmentation_pipeline(class_id)

                # Get all samples of this class
                class_samples = []
                for label_file in train_labels_dir.glob('*.txt'):
                    bboxes = self.process_bbox_file(label_file)
                    if any(cid == class_id for cid, _ in bboxes):
                        class_samples.append(label_file)
                
                if not class_samples:
                    self.logger.warning(f"No samples found for class {class_id}, skipping augmentation")
                    continue
                    
                self.logger.info(f"Found {len(class_samples)} samples for class {class_id}")
                # Augment samples
                aug_per_sample = max(1, num_aug // len(class_samples))
                self.logger.info(f"Will create {aug_per_sample} augmentations per sample for class {class_id}")
                
                for label_file in tqdm(class_samples, desc=f"Class {class_id}"):
                    img_file = self.base_path / 'train/images' / f"{label_file.stem}.jpg"
                    for i in range(aug_per_sample):
                        self.augment_sample(img_file, label_file, class_id, transform, i)

            # Copy data.yaml
            shutil.copy2(self.base_path / 'data.yaml', self.output_path / 'data.yaml')
            
            self.logger.info("Augmentation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during augmentation: {e}")
            raise

def main():
    """Main execution function."""
    # Usar el dataset limpio del paso anterior
    dataset_path = "data/balanced_dataset_backup_20250120_105413"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
    logging.info(f"Using dataset from: {dataset_path}")
    augmentor = LogoAugmentor(dataset_path)
    augmentor.augment_dataset()

if __name__ == "__main__":
    main()