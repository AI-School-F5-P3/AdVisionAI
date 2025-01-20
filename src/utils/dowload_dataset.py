# src/utils/download_dataset.py
import os
import requests
import logging
from pathlib import Path
import zipfile
import shutil

def setup_logging():
   logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_extract():
   # URL directa de Roboflow
   url = "https://app.roboflow.com/ds/nCfl4sPiio?key=VTFocROIVB"
   download_dir = Path('data/roboflow_dataset')
   zip_path = download_dir / 'dataset.zip'

   # Crear directorio si no existe
   download_dir.mkdir(parents=True, exist_ok=True)

   try:
       # Descargar archivo
       logging.info(f"Descargando dataset desde Roboflow...")
       response = requests.get(url, stream=True)
       response.raise_for_status()

       # Guardar archivo
       with open(zip_path, 'wb') as f:
           for chunk in response.iter_content(chunk_size=8192):
               f.write(chunk)
       
       logging.info(f"Dataset descargado en: {zip_path}")
       
       # Extraer contenido
       logging.info("Extrayendo archivos...")
       with zipfile.ZipFile(zip_path, 'r') as zip_ref:
           zip_ref.extractall(download_dir)
       
       # Eliminar zip
       zip_path.unlink()
       
       # Verificar contenido
       files = list(download_dir.glob('**/*'))
       logging.info(f"Total archivos extraídos: {len(files)}")
       for f in files[:10]:  # Mostrar los primeros 10 archivos
           logging.info(f"Encontrado: {f}")
           
   except Exception as e:
       logging.error(f"Error en la descarga: {str(e)}")

if __name__ == "__main__":
   setup_logging()
   download_and_extract()