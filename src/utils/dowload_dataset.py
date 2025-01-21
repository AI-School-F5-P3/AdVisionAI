"""
Dataset augmentation tool with post-transform validation.
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
from typing import Dict, List, Tuple, Optional

class LogoAugmentor:
    def __init__(self, base_path: str = "data/balanced_dataset_backup_20250120_105413"):
        self.base_path = Path(base_path)
        self.output_path = self.base_path.parent / f"augmented_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.setup_logging()
        
        # Target counts for balancing
        self.target_counts = {i: 250 for i in range(7)}  # 250 samples per class
        self.min_bbox_size = 0.01

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

    def validate_bbox(self, bbox: List[float], class_id: int) -> Optional[List[float]]:
        """
        Validate and fix bbox coordinates if possible.
        Returns None if bbox cannot be fixed.
        """
        try:
            x_center, y_center, width, height = bbox
            
            # Fix coordinates to be within [0, 1]
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            
            # Ensure minimum size
            width = max(self.min_bbox_size, min(1-1e-6, width))
            height = max(self.min_bbox_size, min(1-1e-6, height))
            
            # Verify box is within image bounds
            x_min = x_center - width/2
            x_max = x_center + width/2
            y_min = y_center - height/2
            y_max = y_center + height/2
            
            # If box is partially outside, try to fix it
            if x_min < 0:
                x_center = width/2
            elif x_max > 1:
                x_center = 1 - width/2
                
            if y_min < 0:
                y_center = height/2
            elif y_max > 1:
                y_center = 1 - height/2
            
            # Final validation
            x_min = x_center - width/2
            x_max = x_center + width/2
            y_min = y_center - height/2
            y_max = y_center + height/2
            
            if (x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1 or
                x_max <= x_min or y_max <= y_min):
                return None
                
            return [x_center, y_center, width, height]
            
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Error validating bbox: {e}")
            return None

    def get_augmentation_pipeline(self, class_id: int) -> A.Compose:
        """Get class-specific augmentation pipeline."""
        # Base transforms (moderados)
        transforms = [
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.7
            )
        ]
        
        # Augmentación adicional para clases minoritarias
        if class_id in [5, 2]:  # women_secret, desigual
            transforms.extend([
                A.RandomShadow(p=0.3),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            ])
        
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3
            )
        )

    def augment_sample(self, img_path: Path, bbox_path: Path, class_id: int, 
                      transform: A.Compose, output_idx: int) -> bool:
        """
        Augment a single sample with validation.
        Returns True if augmentation was successful.
        """
        try:
            # Read image and bboxes
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            bboxes = []
            class_labels = []
            
            with open(bbox_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:5]]
                            validated_bbox = self.validate_bbox(bbox, cls_id)
                            if validated_bbox:
                                bboxes.append(validated_bbox)
                                class_labels.append(cls_id)
                        except ValueError:
                            continue
            
            if not bboxes:
                return False
            
            # Apply augmentation
            transformed = transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            # Validate transformed bboxes
            valid_bboxes = []
            valid_labels = []
            
            for bbox, label in zip(transformed['bboxes'], transformed['class_labels']):
                validated_bbox = self.validate_bbox(bbox, label)
                if validated_bbox:
                    valid_bboxes.append(validated_bbox)
                    valid_labels.append(label)
            
            if not valid_bboxes:
                return False
            
            # Save augmented image
            aug_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
            aug_img_path = self.output_path / 'train/images' / f'{img_path.stem}_aug_{output_idx}.jpg'
            cv2.imwrite(str(aug_img_path), aug_image)
            
            # Save augmented bboxes
            aug_bbox_path = self.output_path / 'train/labels' / f'{img_path.stem}_aug_{output_idx}.txt'
            with open(aug_bbox_path, 'w') as f:
                for bbox, label in zip(valid_bboxes, valid_labels):
                    bbox_str = ' '.join([f'{x:.6f}' for x in bbox])
                    f.write(f'{label} {bbox_str}\n')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error augmenting {img_path}: {e}")
            return False

    def augment_dataset(self):
        """Execute dataset augmentation with validation."""
        try:
            # Create output structure
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

            # Count existing samples
            train_labels_dir = self.base_path / 'train' / 'labels'
            class_counts = {i: 0 for i in range(7)}
            
            for label_file in train_labels_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    for line in f:
                        try:
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1
                        except (ValueError, IndexError):
                            continue

            self.logger.info(f"Original distribution: {class_counts}")
            
            # Calculate needed augmentations
            aug_needed = {
                class_id: max(0, self.target_counts[class_id] - count)
                for class_id, count in class_counts.items()
            }
            
            self.logger.info(f"Augmentations needed: {aug_needed}")

            # Perform augmentation
            augmentation_stats = {
                'attempted': 0,
                'successful': 0,
                'failed': 0
            }

            for class_id, num_aug in aug_needed.items():
                if num_aug <= 0:
                    continue

                self.logger.info(f"\nAugmenting class {class_id}")
                transform = self.get_augmentation_pipeline(class_id)
                
                # Collect samples for this class
                class_samples = []
                for label_file in train_labels_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        if any(line.startswith(f"{class_id} ") for line in f):
                            class_samples.append(label_file)

                if not class_samples:
                    self.logger.warning(f"No samples found for class {class_id}")
                    continue

                # Calculate augmentations per sample
                aug_per_sample = max(1, num_aug // len(class_samples))
                self.logger.info(f"Will create {aug_per_sample} augmentations per sample")

                for label_file in tqdm(class_samples, desc=f"Class {class_id}"):
                    img_file = self.base_path / 'train' / 'images' / f"{label_file.stem}.jpg"
                    if not img_file.exists():
                        img_file = self.base_path / 'train' / 'images' / f"{label_file.stem}.jpeg"
                    
                    if img_file.exists():
                        for i in range(aug_per_sample):
                            augmentation_stats['attempted'] += 1
                            success = self.augment_sample(img_file, label_file, class_id, transform, i)
                            if success:
                                augmentation_stats['successful'] += 1
                            else:
                                augmentation_stats['failed'] += 1

            # Copy data.yaml
            shutil.copy2(self.base_path / 'data.yaml', self.output_path / 'data.yaml')
            
            self.logger.info("\nAugmentation Statistics:")
            self.logger.info(f"Attempted: {augmentation_stats['attempted']}")
            self.logger.info(f"Successful: {augmentation_stats['successful']}")
            self.logger.info(f"Failed: {augmentation_stats['failed']}")
            
        except Exception as e:
            self.logger.error(f"Error during augmentation: {e}")
            raise

def main():
    """Main execution function."""
    try:
        augmentor = LogoAugmentor()
        augmentor.augment_dataset()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    