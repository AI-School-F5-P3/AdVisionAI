import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
import logging
from concurrent.futures import ProcessPoolExecutor
import argparse

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def create_transform():
    return A.Compose([
        # Primero redimensionar a un tamaño fijo
        A.Resize(height=300, width=300, always_apply=True),
        
        # Luego aplicar las transformaciones
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, p=0.5),
        
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
        ], p=0.5),
        
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
        ], p=0.3),
        
        A.CoarseDropout(
            max_holes=8,
            max_height=20,
            max_width=20,
            min_holes=5,
            min_height=10,
            min_width=10,
            p=0.3
        ),
    ])

def create_augmentations(image_path, output_dir, num_aug=5):
    try:
        # Leer imagen con PIL y convertir a numpy array
        img = np.array(Image.open(str(image_path)))
        if img is None:
            logging.warning(f"No se pudo leer la imagen: {image_path}")
            return []
            
        transform = create_transform()
        augmented_images = []
        
        # Crear augmentaciones
        for i in range(num_aug):
            try:
                transformed = transform(image=img)['image']
                output_path = output_dir / f'{image_path.stem}_aug_{i}.png'
                
                # Guardar imagen usando PIL
                Image.fromarray(transformed).save(output_path)
                augmented_images.append(output_path)
                    
            except Exception as e:
                logging.error(f"Error en augmentación {i} para {image_path}: {str(e)}")
                continue
                
        return augmented_images
        
    except Exception as e:
        logging.error(f"Error al procesar la imagen {image_path}: {str(e)}")
        return []

def process_image(args):
    img_path, output_dir, num_aug = args
    return create_augmentations(img_path, output_dir, num_aug)

def augment_dataset(clean_output_dir=False, num_aug=5):
    setup_logging()
    
    # Preparar directorios
    base_dir = Path('data/processed/corte_ingles')
    train_dir = base_dir / 'train'
    aug_dir = base_dir / 'train_augmented'
    aug_dir.mkdir(exist_ok=True, parents=True)
    
    # Limpiar directorio si es necesario
    if clean_output_dir:
        logging.info(f"Limpiando directorio de salida: {aug_dir}")
        for file in aug_dir.glob('*.*'):
            try:
                file.unlink()
            except Exception as e:
                logging.warning(f"No se pudo eliminar {file}: {e}")
    
    # Copiar imágenes originales
    logging.info("Copiando imágenes originales...")
    for img_path in train_dir.glob('*.*'):
        try:
            Image.open(img_path).save(aug_dir / img_path.name)
        except Exception as e:
            logging.warning(f"No se pudo copiar la imagen {img_path}: {e}")
    
    # Crear augmentaciones
    logging.info("Iniciando proceso de augmentación...")
    total_augmented = 0
    
    # Procesar imágenes en paralelo
    with ProcessPoolExecutor() as executor:
        args = [(img_path, aug_dir, num_aug) for img_path in train_dir.glob('*.*')]
        for result in executor.map(process_image, args):
            total_augmented += len(result)
    
    # Generar informe
    logging.info("=== Informe de Augmentación ===")
    logging.info(f"Imágenes originales: {len(list(train_dir.glob('*.*')))}")
    logging.info(f"Augmentaciones creadas: {total_augmented}")
    logging.info(f"Total imágenes: {len(list(aug_dir.glob('*.*')))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true', help='Limpiar carpeta de salida antes de comenzar')
    parser.add_argument('--num_aug', type=int, default=5, help='Número de augmentaciones por imagen')
    args = parser.parse_args()
    
    augment_dataset(clean_output_dir=args.clean, num_aug=args.num_aug)