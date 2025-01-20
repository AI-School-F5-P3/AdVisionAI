import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

class DatasetBalancer:
    def __init__(self, base_dir="data/roboflow_dataset"):
        self.base_dir = Path(base_dir)
        self.splits = ['train', 'valid', 'test']
        self.class_names = {
            0: 'bershka',
            1: 'corte_ingles',
            2: 'desigual',
            3: 'mango',
            4: 'stradivarius',
            5: 'women_secret',
            6: 'zara'
        }
        
    def collect_all_samples(self):
        """Recolecta todas las muestras, incluyendo negativas"""
        all_samples = []
        negative_samples = []
        
        for split in self.splits:
            images_dir = self.base_dir / split / "images"
            labels_dir = self.base_dir / split / "labels"
            
            print(f"\nProcesando {split}...")
            
            # Recolectar todas las imágenes
            for image_file in images_dir.glob("*.*"):
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    label_file = labels_dir / f"{image_file.stem}.txt"
                    
                    if label_file.exists():
                        try:
                            with open(label_file, 'r') as f:
                                lines = f.readlines()
                                if lines:  # Archivo con etiquetas
                                    classes = [int(line.split()[0]) for line in lines]
                                    all_samples.append({
                                        'label_path': label_file,
                                        'image_path': image_file,
                                        'classes': classes,
                                        'is_negative': False,
                                        'split': split
                                    })
                                else:  # Archivo vacío = muestra negativa
                                    negative_samples.append({
                                        'label_path': label_file,
                                        'image_path': image_file,
                                        'classes': [],
                                        'is_negative': True,
                                        'split': split
                                    })
                        except Exception as e:
                            print(f"Error procesando {label_file}: {e}")
                    else:  # No existe archivo de etiquetas = muestra negativa
                        # Crear archivo de etiquetas vacío
                        negative_samples.append({
                            'label_path': label_file,
                            'image_path': image_file,
                            'classes': [],
                            'is_negative': True,
                            'split': split
                        })
        
        print(f"\nTotal muestras positivas: {len(all_samples)}")
        print(f"Total muestras negativas: {len(negative_samples)}")
        
        return all_samples, negative_samples
    
    def redistribute_samples(self, positive_samples, negative_samples, train_size=0.8, val_size=0.1):
        """Redistribuye las muestras manteniendo balance de clases y negativos"""
        print("\nDistribución actual por clase:")
        class_samples = {i: [] for i in range(len(self.class_names))}
        for sample in positive_samples:
            if sample['classes']:
                main_class = sample['classes'][0]
                class_samples[main_class].append(sample)
        
        for class_id, samples_list in class_samples.items():
            print(f"{self.class_names[class_id]}: {len(samples_list)}")
        print(f"Negativos: {len(negative_samples)}")
        
        # Redistribuir positivos por clase
        train_samples, val_samples, test_samples = [], [], []
        for class_id, class_data in class_samples.items():
            if not class_data:
                print(f"\nAdvertencia: No hay muestras para {self.class_names[class_id]}")
                continue
                
            # Split train/temp
            class_train, class_temp = train_test_split(
                class_data, train_size=train_size, random_state=42
            )
            # Split valid/test
            val_ratio = val_size / (1 - train_size)
            class_val, class_test = train_test_split(
                class_temp, train_size=val_ratio, random_state=42
            )
            
            train_samples.extend(class_train)
            val_samples.extend(class_val)
            test_samples.extend(class_test)
        
        # Redistribuir negativos en la misma proporción
        if negative_samples:
            neg_train, neg_temp = train_test_split(
                negative_samples, train_size=train_size, random_state=42
            )
            neg_val, neg_test = train_test_split(
                neg_temp, train_size=val_ratio, random_state=42
            )
            
            train_samples.extend(neg_train)
            val_samples.extend(neg_val)
            test_samples.extend(neg_test)
        
        return train_samples, val_samples, test_samples
    
    def save_redistributed_dataset(self, train_samples, val_samples, test_samples):
        """Guarda el dataset redistribuido"""
        output_base = self.base_dir.parent / "balanced_dataset"
        
        # Crear directorios
        for split in self.splits:
            for subdir in ['images', 'labels']:
                (output_base / split / subdir).mkdir(parents=True, exist_ok=True)
        
        def copy_samples(samples, split):
            for sample in samples:
                try:
                    # Copiar imagen
                    dst_img = output_base / split / "images" / sample['image_path'].name
                    shutil.copy2(sample['image_path'], dst_img)
                    
                    # Para muestras negativas, crear archivo de etiquetas vacío
                    dst_label = output_base / split / "labels" / f"{sample['image_path'].stem}.txt"
                    if sample['is_negative']:
                        # Crear archivo vacío
                        dst_label.touch()
                    else:
                        # Copiar archivo de etiquetas existente
                        shutil.copy2(sample['label_path'], dst_label)
                        
                except Exception as e:
                    print(f"Error copiando {sample['image_path']}: {e}")
        
        print("\nGuardando muestras redistribuidas...")
        copy_samples(train_samples, "train")
        copy_samples(val_samples, "valid")
        copy_samples(test_samples, "test")
        
        # Crear nuevo data.yaml
        yaml_content = f"""train: train/images
val: valid/images
test: test/images

names:
  0: bershka
  1: corte_ingles
  2: desigual
  3: mango
  4: stradivarius
  5: women_secret
  6: zara"""
        
        with open(output_base / "data.yaml", 'w') as f:
            f.write(yaml_content)
        
        return output_base

    def print_distribution(self, train_samples, val_samples, test_samples):
        """Imprime la distribución de clases y negativos en cada split"""
        def analyze_split(samples):
            counts = {i: 0 for i in range(len(self.class_names))}
            negatives = 0
            for sample in samples:
                if sample['is_negative']:
                    negatives += 1
                elif sample['classes']:
                    main_class = sample['classes'][0]
                    counts[main_class] += 1
            return counts, negatives
        
        print("\nDistribución del dataset balanceado:")
        
        for split_name, split_samples in [
            ("Train", train_samples),
            ("Validation", val_samples),
            ("Test", test_samples)
        ]:
            counts, negatives = analyze_split(split_samples)
            print(f"\n{split_name}:")
            for class_id, count in counts.items():
                print(f"{self.class_names[class_id]}: {count}")
            print(f"Negativos: {negatives}")

def main():
    print("Iniciando proceso de balanceo del dataset...")
    balancer = DatasetBalancer()
    
    # Recolectar muestras
    print("Recolectando muestras...")
    positive_samples, negative_samples = balancer.collect_all_samples()
    
    if not positive_samples and not negative_samples:
        print("No se encontraron muestras")
        return
    
    # Redistribuir muestras
    print("\nRedistribuyendo muestras...")
    train_samples, val_samples, test_samples = balancer.redistribute_samples(
        positive_samples, negative_samples
    )
    
    # Guardar dataset balanceado
    print("\nGuardando dataset balanceado...")
    output_dir = balancer.save_redistributed_dataset(train_samples, val_samples, test_samples)
    
    # Imprimir distribución final
    balancer.print_distribution(train_samples, val_samples, test_samples)
    
    print(f"\nDataset balanceado guardado en: {output_dir}")

if __name__ == "__main__":
    main()