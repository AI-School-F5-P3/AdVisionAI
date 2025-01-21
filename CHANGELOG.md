# Changelog

## Visión General
Este archivo documenta todos los cambios notables en el proyecto AdVisionAI. El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [3.0.0] - 2025-01-21
### Mejorado
- Rendimiento general del modelo:
  - mAP50 mejorado de 0.623 a 0.816 (+31%)
  - mAP50-95 mejorado de 0.332 a 0.439 (+32%)
  - Rendimiento por marca:
    - Zara: 0.961 mAP50
    - Corte Inglés: 0.934 mAP50
    - Mango: 0.874 mAP50
    - Stradivarius: 0.844 mAP50
    - Women Secret: 0.763 mAP50
    - Desigual: 0.729 mAP50
    - Bershka: 0.609 mAP50
- Velocidad de inferencia optimizada:
  - Preprocesamiento: 0.1ms
  - Inferencia: 1.2ms
  - Postprocesamiento: 2.2ms

### Añadido
- Sistema de balanceo de clases con pesos normalizados
- Pipeline de augmentación mejorado
- Hiperparámetros optimizados para fine-tuning
- Early stopping con paciencia extendida
- Herramientas de validación y limpieza de dataset

### Corregido
- Problemas con coordenadas no normalizadas en etiquetas
- Desbalance en el dataset de entrenamiento
- Inestabilidad durante el entrenamiento

## [2.0.0] - 2025-01-20
### Añadido
- Primera versión del modelo multi-marca
- Soporte para 7 marcas diferentes
- Dataset balanceado inicial

## [1.0.0] - 2025-01-15
### Añadido
- Modelo inicial para detección de Corte Inglés
- Pipeline básico de entrenamiento
- Sistema de evaluación de modelos