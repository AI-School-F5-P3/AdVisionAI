import os
import shutil
from PIL import Image
from pathlib import Path

def process_images(source_dir, dest_dir):
    # Crear el directorio si no existe
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    
    # Lista para almacenar información de las imágenes
    image_info = []
    
    # Procesar cada imagen
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(source_dir, filename)
            try:
                # Abrir imagen y obtener información
                with Image.open(img_path) as img:
                    width, height = img.size
                    format = img.format
                    image_info.append({
                        'filename': filename,
                        'width': width,
                        'height': height,
                        'format': format
                    })
                
                # Copiar imagen al nuevo directorio
                new_filename = f'corte_ingles_{len(image_info):04d}.{format.lower()}'
                shutil.copy2(img_path, os.path.join(dest_dir, new_filename))
                
            except Exception as e:
                print(f'Error procesando {filename}: {str(e)}')
    
    # Generar informe
    print('\nResumen de las imágenes:')
    print(f'Total de imágenes: {len(image_info)}')
    
    # Mostrar dimensiones únicas
    dimensions = set((info['width'], info['height']) for info in image_info)
    print(f'\nDimensiones encontradas:')
    for width, height in sorted(dimensions):
        count = sum(1 for info in image_info if info['width'] == width and info['height'] == height)
        print(f'{width}x{height}: {count} imágenes')

if __name__ == '__main__':
    source_dir = r'C:\Users\samir\OneDrive\Escritorio\Corte ingles'
    dest_dir = r'C:\Users\samir\AdVisionAI\data\raw\corte_ingles\images'
    
    process_images(source_dir, dest_dir)
