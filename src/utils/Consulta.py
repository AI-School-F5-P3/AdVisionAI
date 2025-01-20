import os

def analyze_split(split_dir):
    class_counts = {i: 0 for i in range(7)}  # 7 clases (0-6)
    for label_file in os.listdir(split_dir):
        with open(os.path.join(split_dir, label_file)) as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
    return class_counts

# Ruta del directorio de etiquetas de la división de entrenamiento
train_labels_dir = "data/roboflow_dataset/train/labels"
train_class_counts = analyze_split(train_labels_dir)
print("Distribución de clases en el conjunto de entrenamiento:", train_class_counts)

# Ruta del directorio de etiquetas de la división de validación
valid_labels_dir = "data/roboflow_dataset/valid/labels"
valid_class_counts = analyze_split(valid_labels_dir)
print("Distribución de clases en el conjunto de validación:", valid_class_counts)

# Ruta del directorio de etiquetas de la división de prueba
test_labels_dir = "data/roboflow_dataset/test/labels"
test_class_counts = analyze_split(test_labels_dir)
print("Distribución de clases en el conjunto de prueba:", test_class_counts)
