# Modificar verify_model.py para ver más detalles
from ultralytics import YOLO
from pathlib import Path
import cv2

def verify_model():
    model_path = Path('models/trained/v5/best.pt')
    model = YOLO(model_path)
    
    print("\nInformación del modelo:")
    print(f"Tipo de modelo: {model.task}")
    print(f"Nombres de clases: {model.names}")
    print(f"Modelo cargado desde: {model_path.absolute()}")
    
    # Cargar y mostrar configuración del entrenamiento
    if Path('config/data.yaml').exists():
        import yaml
        with open('config/data.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
        print("\nConfiguración del dataset:")
        print(f"Clases configuradas: {data_config.get('names', [])}")
    
    # Probar predicción con más detalle
    test_image = "data/test/images/descargar.jpeg"
    results = model.predict(test_image, 
                          conf=0.01,  # Umbral muy bajo para ver cualquier detección
                          verbose=True)
    
    for r in results:
        print(f"\nAnálisis detallado:")
        boxes = r.boxes
        print(f"- Detecciones totales (conf > 0.01): {len(boxes)}")
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                print(f"- Detección {i+1}:")
                print(f"  Confianza: {box.conf[0].item():.3f}")
                print(f"  Clase: {model.names[int(box.cls[0])]}")

if __name__ == "__main__":
    verify_model()