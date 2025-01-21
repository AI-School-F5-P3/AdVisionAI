import os
import zipfile
from pathlib import Path
from datetime import datetime
import shutil

def import_models():
    """Importa versiones del modelo a la estructura del proyecto."""
    # Configurar paths
    models = {
        'v2': r"C:\Users\samir\Downloads\improved_model_20250120_021056.zip",
        'v3': r"C:\Users\samir\Downloads\improved_model_v3_20250121_002007.zip"
    }
    
    project_root = Path(os.getcwd())
    models_dir = project_root / 'models' / 'trained'
    docs_dir = project_root / 'docs' / 'models'

    for version, zip_path in models.items():
        try:
            print(f"\nImportando modelo {version} desde: {zip_path}")
            
            # Crear directorio específico para la versión
            version_dir = models_dir / version
            os.makedirs(version_dir, exist_ok=True)
            
            # Extraer archivos
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extraer modelo
                for file in zip_ref.namelist():
                    if file.endswith('.pt'):
                        zip_ref.extract(file, version_dir)
                        # Renombrar si es necesario
                        extracted_path = version_dir / file
                        if 'best' in file:
                            shutil.move(extracted_path, version_dir / 'best.pt')
                        elif 'last' in file:
                            shutil.move(extracted_path, version_dir / 'last.pt')
                        print(f"Modelo guardado: {file}")
                    
                    # Copiar métricas y gráficos con prefijo de versión
                    elif file.endswith(('.png', '.txt')):
                        base_name = os.path.basename(file)
                        new_name = f"{version}_{base_name}"
                        zip_ref.extract(file, docs_dir)
                        os.rename(docs_dir / file, docs_dir / new_name)
                        print(f"Documentación guardada: {new_name}")

            print(f"Modelo {version} importado exitosamente!")
            
        except Exception as e:
            print(f"\nError importando modelo {version}: {e}")

if __name__ == "__main__":
    import_models()