import os
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def standardize_image(img, target_size=(300, 300)):
    # Crear nueva imagen con fondo negro
    new_img = Image.new('RGB', target_size, (0, 0, 0))
    
    # Calcular nueva dimensión manteniendo aspect ratio
    ratio = min(target_size[0]/img.size[0], target_size[1]/img.size[1])
    new_size = tuple([int(x*ratio) for x in img.size])
    
    # Redimensionar imagen
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Calcular posición para centrar
    pos = ((target_size[0] - new_size[0])//2, (target_size[1] - new_size[1])//2)
    
    # Pegar imagen centrada
    new_img.paste(img, pos)
    return new_img

def process_dataset(dataset_type='positive'):
    setup_logging()
    
    # Configurar rutas según el tipo de dataset
    if dataset_type == 'positive':
        raw_dir = Path('data/raw/corte_ingles/images')
        processed_dir = Path('data/processed/corte_ingles')
        prefix = 'corte_ingles'
    else:
        raw_dir = Path(r"C:\Users\samir\OneDrive\Escritorio\Negativas")
        processed_dir = Path('data/processed/negative')
        prefix = 'negative'
    
    logging.info(f"Procesando imágenes {dataset_type}")
    logging.info(f"Directorio origen: {raw_dir}")
    logging.info(f"Directorio destino: {processed_dir}")
    
    # Crear directorios si no existen
    for dir_name in ['train', 'val', 'test']:
        (processed_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Procesar cada imagen
    images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        images.extend(list(raw_dir.glob(ext)))
    
    if not images:
        logging.error(f"No se encontraron imágenes en {raw_dir}")
        return
    
    # Dividir en train (70%), val (20%), test (10%)
    np.random.shuffle(images)
    n_train = int(0.7 * len(images))
    n_val = int(0.2 * len(images))
    
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train+n_val]
    test_imgs = images[n_train+n_val:]
    
    # Procesar cada conjunto
    for img_path, subset in zip(
        [train_imgs, val_imgs, test_imgs],
        ['train', 'val', 'test']
    ):
        logging.info(f'Procesando conjunto {subset}...')
        for i, img_file in enumerate(img_path):
            try:
                save_path = processed_dir / subset / f'{prefix}_{subset}_{i:04d}.png'
                
                # Verificar si ya existe
                if save_path.exists():
                    logging.info(f"Saltando {save_path.name} - ya existe")
                    continue
                
                with Image.open(img_file) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    std_img = standardize_image(img)
                    std_img.save(save_path, 'PNG')
                    
            except Exception as e:
                logging.error(f"Error procesando {img_file}: {str(e)}")
                continue
    
    logging.info('\nResumen:')
    logging.info(f'Train: {len(train_imgs)} imágenes')
    logging.info(f'Val: {len(val_imgs)} imágenes')
    logging.info(f'Test: {len(test_imgs)} imágenes')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['positive', 'negative'], 
                       default='positive',
                       help='Tipo de imágenes a procesar (positive/negative)')
    args = parser.parse_args()
    
    process_dataset(args.type)