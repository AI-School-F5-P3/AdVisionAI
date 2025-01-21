"""
Fix coordinates that are out of range in label files.
"""

import os
from pathlib import Path
import logging
import numpy as np

def fix_coordinates(coords):
    """
    Fix coordinates to be within [0,1] range while maintaining aspect ratio.
    """
    x_center, y_center, width, height = coords
    
    # Clip centers to valid range
    x_center = float(np.clip(x_center, width/2, 1-width/2))
    y_center = float(np.clip(y_center, height/2, 1-height/2))
    
    # Clip dimensions if needed
    width = float(min(width, 1.0))
    height = float(min(height, 1.0))
    
    # Ensure the box fits within image bounds
    x_min = x_center - width/2
    x_max = x_center + width/2
    y_min = y_center - height/2
    y_max = y_center + height/2
    
    if x_min < 0:
        x_center += abs(x_min)
    if x_max > 1:
        x_center -= (x_max - 1)
    if y_min < 0:
        y_center += abs(y_min)
    if y_max > 1:
        y_center -= (y_max - 1)
    
    return [float(x_center), float(y_center), float(width), float(height)]

def fix_specific_files():
    """Fix the specific files with coordinate issues."""
    problem_files = [
        "data/augmented_dataset_20250120_131507/test/labels/raw_img_0397_png.rf.7bf36078f39cd6ddd4d3f2c7bbf1c318.txt",
        "data/augmented_dataset_20250120_131507/test/labels/raw_img_0413_png.rf.5f624268f1a9e6210629efefde42b3d3.txt",
        "data/augmented_dataset_20250120_131507/test/labels/raw_img_0431_png.rf.727dd47189d9f40a3d3eac7b0a408c39.txt",
        "data/augmented_dataset_20250120_131507/test/labels/raw_img_0775_png.rf.f6039a23be3de308ceddfe2f9c9492f8.txt"
    ]
    
    for file_path in problem_files:
        try:
            print(f"\nProcessing: {file_path}")
            file_path = Path(file_path)
            
            if not file_path.exists():
                print(f"File not found: {file_path}")
                continue
            
            # Read and fix file
            fixed_lines = []
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(float(parts[0]))
                        coords = [float(x) for x in parts[1:5]]
                        
                        # Fix coordinates
                        fixed_coords = fix_coordinates(coords)
                        
                        # Format fixed line
                        fixed_line = f"{class_id} {' '.join(f'{x:.6f}' for x in fixed_coords)}"
                        fixed_lines.append(fixed_line)
                        
                        print(f"Original coords: {coords}")
                        print(f"Fixed coords: {fixed_coords}")
            
            # Save fixed content
            if fixed_lines:
                with open(file_path, 'w') as f:
                    f.write('\n'.join(fixed_lines) + '\n')
                print(f"Fixed and saved: {file_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    fix_specific_files()
    print("\nCoordinate fixing completed!")