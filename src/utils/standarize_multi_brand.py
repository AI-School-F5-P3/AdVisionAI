# src/utils/standardize_multi_brand.py
import os
from PIL import Image
import shutil
from pathlib import Path
import logging

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def standardize_image(img, target_size=(300, 300)):
    # Convertir a RGB si es necesario
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Obtener dimensiones actuales
    width, height = img.size
    
    # Calcular nueva dimensión manteniendo aspect ratio
    ratio = min(float(target_size[0])/width, float(target_size[1])/height)
    new_size = (int(width * ratio), int(height * ratio))
    
    # Redimensionar imagen
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img

def process_raw_images(source_dir: str, output_base_dir: str):
    setup_logging()
    source_path = Path(source_dir)
    output_path = Path(output_base_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extensiones válidas
    valid_extensions = {'.jpg', '.jpeg', '.png', '.svg'}
    
    # Procesar cada imagen
    total_processed = 0
    errors = 0
    
    logging.info(f"Buscando imágenes en: {source_path}")
    
    for img_file in source_path.iterdir():
        if img_file.suffix.lower() in valid_extensions:
            try:
                logging.info(f"Procesando: {img_file.name}")
                
                # Abrir imagen
                with Image.open(img_file) as img:
                    # Estandarizar imagen
                    std_img = standardize_image(img)
                    
                    # Crear nombre de archivo de salida
                    output_file = output_path / f'raw_img_{total_processed:04d}.png'
                    
                    # Guardar imagen procesada
                    std_img.save(output_file, 'PNG')
                    total_processed += 1
                    
                    if total_processed % 10 == 0:
                        logging.info(f"Procesadas {total_processed} imágenes...")
                        
            except Exception as e:
                logging.error(f"Error procesando {img_file}: {str(e)}")
                errors += 1
                continue
    
    logging.info(f"\nResumen del proceso:")
    logging.info(f"Total imágenes procesadas exitosamente: {total_processed}")
    logging.info(f"Total errores encontrados: {errors}")

if __name__ == "__main__":
    # Configurar rutas
    source_dir = r"C:\Users\samir\OneDrive\Escritorio\Mango"
    output_dir = "data/raw/pending_labeling"
    
    # Procesar imágenes
    process_raw_images(source_dir, output_dir)