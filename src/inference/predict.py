# En src/inference/predict.py
from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def load_model():
    """Carga el modelo entrenado."""
    model_path = Path('models/trained/best.pt')
    return YOLO(model_path)

def predict_image(model, image_path):
    """Realiza predicción en una imagen."""
    results = model.predict(image_path)
    return results[0]  # Retorna el primer resultado

def main():
    # Cargar modelo
    model = load_model()
    
    # Realizar predicción en una imagen de prueba
    image_path = 'C:/Users/samir/AdVisionAI/src/inference/data/test/images'
    results = predict_image(model, image_path)
    
    # Mostrar resultados
    print(f"Detecciones encontradas: {len(results.boxes)}")
    for box in results.boxes:
        confidence = box.conf[0].item()
        print(f"Confianza: {confidence:.2f}")

if __name__ == "__main__":
    main()
