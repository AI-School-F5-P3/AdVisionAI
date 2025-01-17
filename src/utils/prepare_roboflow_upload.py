import os
import shutil
from pathlib import Path
import zipfile

def prepare_roboflow_upload():
    # Directorio con las imágenes de entrenamiento
    source_dir = Path('data/processed/corte_ingles/train_augmented')
    # Directorio para el archivo zip
    zip_path = Path('data/roboflow_upload.zip')
    
    # Crear zip con las imágenes
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for img_path in source_dir.glob('*.png'):
            zipf.write(img_path, img_path.name)
    
    print(f'Archivo zip creado en: {zip_path}')
    print(f'Total de imágenes: {len(list(source_dir.glob("*.png")))}')

if __name__ == '__main__':
    prepare_roboflow_upload()
