"""
Fix and analyze label files in detail.
"""

import os
from pathlib import Path
import logging
from collections import defaultdict

class LabelAnalyzer:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_and_fix_labels(self):
        """Analyze and fix label files with detailed reporting."""
        stats = {
            'total_files': 0,
            'processed_files': 0,
            'files_with_issues': [],
            'class_distribution': defaultdict(int),
            'format_issues': []
        }
        
        for split in ['train', 'valid', 'test']:
            labels_dir = self.dataset_path / split / 'labels'
            if not labels_dir.exists():
                self.logger.warning(f"Directory not found: {labels_dir}")
                continue
                
            self.logger.info(f"\nAnalyzing {split} split...")
            files = list(labels_dir.glob('*.txt'))
            stats['total_files'] += len(files)
            
            for label_file in files:
                stats['processed_files'] += 1
                try:
                    needs_fix = False
                    fixed_lines = []
                    problematic_lines = []
                    
                    with open(label_file, 'r') as f:
                        content = f.read()
                        if not content.strip():
                            self.logger.warning(f"Empty file: {label_file}")
                            continue
                            
                        for line_num, line in enumerate(content.splitlines(), 1):
                            parts = line.strip().split()
                            
                            if len(parts) < 5:
                                problematic_lines.append(f"Line {line_num}: Insufficient parts - {line}")
                                continue
                                
                            try:
                                # Intentar convertir y validar cada parte
                                class_id_str = parts[0]
                                class_id = float(class_id_str)
                                coords = [float(x) for x in parts[1:5]]
                                
                                # Verificar si class_id tiene decimales
                                if '.' in class_id_str:
                                    needs_fix = True
                                    class_id = int(class_id)
                                    problematic_lines.append(f"Line {line_num}: Float class_id - {class_id_str}")
                                
                                # Verificar rango de class_id
                                if not (0 <= class_id <= 6):
                                    problematic_lines.append(f"Line {line_num}: Invalid class_id range - {class_id}")
                                    continue
                                
                                # Verificar coordenadas
                                if not all(0 <= x <= 1 for x in coords):
                                    problematic_lines.append(f"Line {line_num}: Coordinates out of range - {coords}")
                                    continue
                                
                                # Añadir línea válida
                                fixed_line = f"{int(class_id)} {' '.join(f'{x:.6f}' for x in coords)}"
                                fixed_lines.append(fixed_line)
                                stats['class_distribution'][int(class_id)] += 1
                                
                            except ValueError as e:
                                problematic_lines.append(f"Line {line_num}: Value error - {str(e)} in {line}")
                                needs_fix = True
                    
                    if problematic_lines:
                        stats['files_with_issues'].append({
                            'file': str(label_file),
                            'issues': problematic_lines
                        })
                    
                    if needs_fix and fixed_lines:
                        # Backup original file
                        backup_path = label_file.with_suffix('.txt.bak')
                        if not backup_path.exists():
                            with open(backup_path, 'w') as f:
                                f.write(content)
                        
                        # Write fixed content
                        with open(label_file, 'w') as f:
                            f.write('\n'.join(fixed_lines) + '\n')
                        self.logger.info(f"Fixed file: {label_file}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {label_file}: {e}")
                    stats['format_issues'].append(f"{label_file}: {str(e)}")
        
        # Print summary
        self.logger.info("\nAnalysis Summary:")
        self.logger.info("-" * 50)
        self.logger.info(f"Total files processed: {stats['total_files']}")
        self.logger.info(f"Files with issues: {len(stats['files_with_issues'])}")
        
        if stats['files_with_issues']:
            self.logger.info("\nDetailed Issues:")
            for file_issue in stats['files_with_issues']:
                self.logger.info(f"\nFile: {file_issue['file']}")
                for issue in file_issue['issues']:
                    self.logger.info(f"  {issue}")
        
        self.logger.info("\nClass Distribution:")
        total_instances = sum(stats['class_distribution'].values())
        for class_id, count in sorted(stats['class_distribution'].items()):
            percentage = (count / total_instances * 100) if total_instances > 0 else 0
            self.logger.info(f"Class {class_id}: {count} ({percentage:.1f}%)")
        
        return stats

def main():
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
    print(f"Processing dataset: {latest_dataset}")
    
    analyzer = LabelAnalyzer(latest_dataset)
    analyzer.analyze_and_fix_labels()

if __name__ == "__main__":
    main()