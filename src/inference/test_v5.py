# src/inference/test_v5.py
from ultralytics import YOLO
import cv2
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)

class LogoDetector:
    def __init__(self, model_path='models/trained/v5/best.pt'):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        """Cargar el modelo YOLO."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en: {self.model_path}")
        model = YOLO(self.model_path)
        logging.info(f"Modelo cargado desde: {self.model_path}")
        return model

    def process_images(self, test_dir='data/test/images', conf_threshold=0.25):
        """Procesar todas las imágenes en el directorio."""
        test_dir = Path(test_dir)
        logging.info(f"Buscando imágenes en: {test_dir.absolute()}")
        
        if not test_dir.exists():
            raise FileNotFoundError(f"No se encontró el directorio de pruebas: {test_dir}")

        # Listar archivos
        self._list_directory_contents(test_dir)
        
        # Procesar imágenes
        images = self._get_image_paths(test_dir)
        
        # Crear directorio para resultados visuales
        output_dir = Path('results/v5_predictions_visual')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Procesar cada imagen
        for img_path in images:
            self._process_single_image(img_path, conf_threshold, output_dir)

    def _list_directory_contents(self, directory):
        """Listar contenido del directorio."""
        logging.info("Archivos encontrados en el directorio:")
        for file in os.listdir(directory):
            logging.info(f"- {file}")

    def _get_image_paths(self, directory):
        """Obtener paths de todas las imágenes."""
        patterns = ['*.jpg', '*.jpeg', '*.png']
        images = []
        for pattern in patterns:
            found = list(directory.glob(pattern))
            logging.info(f"Imágenes {pattern}: {len(found)}")
            images.extend(found)
        logging.info(f"Total imágenes encontradas: {len(images)}")
        return images

    def _process_single_image(self, img_path, conf_threshold, output_dir):
        """Procesar una imagen individual."""
        logging.info(f"\nProcesando: {img_path}")
        
        # Predicción
        results = self.model.predict(
            source=str(img_path),
            conf=conf_threshold,
            save=True,
            save_conf=True,
            save_txt=True
        )

        # Visualizar resultados
        self._visualize_results(img_path, results[0], output_dir)

    def _visualize_results(self, img_path, result, output_dir):
        """Visualizar y guardar resultados."""
        img = cv2.imread(str(img_path))
        boxes = result.boxes
        
        logging.info(f"Detecciones: {len(boxes)}")
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Dibujar bbox y confianza
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'CI: {conf:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            logging.info(f"Confianza: {conf:.3f}")
            logging.info(f"Coordenadas: {[x1, y1, x2, y2]}")

        # Guardar imagen con visualizaciones
        output_path = output_dir / f'vis_{img_path.name}'
        cv2.imwrite(str(output_path), img)

def main():
    try:
        detector = LogoDetector()
        detector.process_images(conf_threshold=0.25)
    except Exception as e:
        logging.error(f"Error durante la ejecución: {str(e)}")

if __name__ == "__main__":
    main()