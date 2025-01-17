import shutil
from pathlib import Path

def prepare_yolo_dataset():
    # Directorio fuente de imágenes positivas aumentadas
    source_dir = Path('data/processed/corte_ingles/train_augmented')
    
    # Directorios destino
    train_dir = Path('data/yolo/images/train')
    val_dir = Path('data/yolo/images/val')
    test_dir = Path('data/yolo/images/test')
    
    # Crear directorios si no existen
    for dir in [train_dir, val_dir, test_dir]:
        dir.mkdir(parents=True, exist_ok=True)
    
    # Copiar imágenes a los directorios correspondientes
    for img_path in source_dir.glob('*.png'):
        # Por ahora, todas las imágenes van a train
        shutil.copy2(img_path, train_dir / img_path.name)

if __name__ == '__main__':
    prepare_yolo_dataset()
