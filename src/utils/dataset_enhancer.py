"""
Dataset Enhancement Tool for AdVisionAI.

This module provides functionality for:
- Dataset validation and cleaning
- Image quality assessment
- Duplicate detection
- Image standardization
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from PIL import Image
import hashlib
from tqdm import tqdm
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Set

class DatasetEnhancer:
    """Class for enhancing and validating the logo detection dataset."""
    
    def __init__(self, base_path: str = "data/balanced_dataset"):
        """
        Initialize the DatasetEnhancer.

        Args:
            base_path: Path to the dataset directory
        """
        self.base_path = Path(base_path)
        self.splits = ['train', 'valid', 'test']
        self.class_names = {
            0: 'bershka',
            1: 'corte_ingles',
            2: 'desigual',
            3: 'mango',
            4: 'stradivarius',
            5: 'women_secret',
            6: 'zara'
        }
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the enhancement process."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs/enhancement")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"enhancement_{timestamp}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_bbox(self, bbox: List[float], img_width: int, img_height: int) -> bool:
        """
        Validate if a bounding box is valid.

        Args:
            bbox: List of [x_center, y_center, width, height]
            img_width: Width of the image
            img_height: Height of the image

        Returns:
            bool: True if bbox is valid
        """
        x_center, y_center, width, height = bbox
        
        # Check normalized format (0-1)
        if not all(0 <= x <= 1 for x in [x_center, y_center, width, height]):
            return False
        
        # Check minimum size
        min_size = 0.01
        if width < min_size or height < min_size:
            return False
        
        # Check if bbox is within image bounds
        if (x_center - width/2 < 0 or x_center + width/2 > 1 or
            y_center - height/2 < 0 or y_center + height/2 > 1):
            return False
            
        return True

    def check_image_quality(self, img_path: Path) -> Tuple[bool, str]:
        """
        Check the quality of an image.

        Args:
            img_path: Path to the image file

        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        try:
            img = Image.open(img_path)
            
            # Format validation
            if img.format not in ['JPEG', 'PNG']:
                return False, f"Unsupported format: {img.format}"
            
            # Size validation
            if any(dim < 100 for dim in img.size):
                return False, f"Image too small: {img.size}"
            
            # Channel validation
            if img.mode not in ['RGB', 'L']:
                return False, f"Unsupported color mode: {img.mode}"
            
            # Content validation
            img_array = np.array(img)
            if img_array.std() < 20:
                return False, "Low variation in image content"
                
            return True, "OK"
            
        except Exception as e:
            return False, f"Error processing image: {e}"

    def clean_dataset(self) -> Dict[str, int]:
        """
        Clean the dataset by removing invalid images and labels.

        Returns:
            Dict with cleaning statistics
        """
        stats = {
            'processed': 0,
            'removed_images': 0,
            'removed_labels': 0,
            'fixed_labels': 0
        }
        
        for split in self.splits:
            images_dir = self.base_path / split / 'images'
            labels_dir = self.base_path / split / 'labels'
            
            for img_path in tqdm(list(images_dir.glob('*')), desc=f'Processing {split}'):
                stats['processed'] += 1
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                # Image quality check
                quality_ok, quality_msg = self.check_image_quality(img_path)
                if not quality_ok:
                    self.logger.warning(f"{img_path}: {quality_msg}")
                    img_path.unlink()
                    if label_path.exists():
                        label_path.unlink()
                    stats['removed_images'] += 1
                    continue
                
                # Label validation
                if label_path.exists():
                    valid_lines = []
                    fixed = False
                    
                    with open(label_path, 'r') as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                                
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:]]
                            
                            if (class_id in self.class_names and 
                                self.validate_bbox(bbox, *Image.open(img_path).size)):
                                valid_lines.append(line)
                            else:
                                fixed = True
                    
                    if fixed:
                        stats['fixed_labels'] += 1
                        with open(label_path, 'w') as f:
                            f.writelines(valid_lines)
                else:
                    stats['removed_labels'] += 1
        
        self.logger.info(f"Cleaning statistics: {stats}")
        return stats

    def detect_duplicates(self) -> Set[Path]:
        """
        Detect duplicate images using MD5 hashing.

        Returns:
            Set of paths to duplicate images
        """
        hashes = {}
        duplicates = set()
        
        for split in self.splits:
            images_dir = self.base_path / split / 'images'
            
            for img_path in tqdm(list(images_dir.glob('*')), desc=f'Checking duplicates in {split}'):
                try:
                    with open(img_path, 'rb') as f:
                        img_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if img_hash in hashes:
                        duplicates.add(img_path)
                    else:
                        hashes[img_hash] = img_path
                        
                except Exception as e:
                    self.logger.error(f"Error processing {img_path}: {e}")
        
        return duplicates

    def standardize_images(self, target_size: Tuple[int, int] = (300, 300)):
        """
        Standardize all images to the target size.

        Args:
            target_size: Desired image dimensions (width, height)
        """
        for split in self.splits:
            images_dir = self.base_path / split / 'images'
            
            for img_path in tqdm(list(images_dir.glob('*')), desc=f'Standardizing {split}'):
                try:
                    img = Image.open(img_path)
                    if img.size != target_size:
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                        img.save(img_path, quality=95)
                except Exception as e:
                    self.logger.error(f"Error standardizing {img_path}: {e}")

    def enhance_dataset(self):
        """Execute all enhancement phases."""
        try:
            # Create backup
            backup_dir = self.base_path.parent / f"{self.base_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.info(f"Creating backup at {backup_dir}")
            shutil.copytree(self.base_path, backup_dir)
            
            # Phase 1: Cleaning
            self.logger.info("Starting cleaning phase...")
            stats = self.clean_dataset()
            
            # Phase 2: Duplicate detection
            self.logger.info("Checking for duplicates...")
            duplicates = self.detect_duplicates()
            for dup in duplicates:
                self.logger.info(f"Removing duplicate: {dup}")
                dup.unlink()
                
            # Phase 3: Standardization
            self.logger.info("Standardizing images...")
            self.standardize_images()
            
            self.logger.info("Dataset enhancement completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during dataset enhancement: {e}")
            raise

def main():
    """Main execution function."""
    enhancer = DatasetEnhancer()
    enhancer.enhance_dataset()

if __name__ == "__main__":
    main()