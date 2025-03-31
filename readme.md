# Proyecto: Predicción de Incidencias con GRU


## Estructura del Proyecto
```
GRU_Prediccion_Incidencias/
|
├── data/              # Datos originales (Excel)
│   └── EXCELINCIDENCIAS.xlsx
|
├── scripts/           # Códigos de entrenamiento y predicción
│   ├── train_gru.py
│   └── predict.py
|
├── models/            # Modelo entrenado y transformadores
│   ├── gru_model/     # Modelo de TensorFlow
│   ├── encoder_causa.pkl
│   └── scaler_y.pkl
|
└── outputs/           # Resultados, gráficas o logs (opcional)

```

---

## Entrenamiento del Modelo

El archivo `scripts/train_gru.py` entrena una red MLP con los siguientes datos:
- **Input**: fecha, y causa 
- **Output**:
  - Número de incidencias
  - Número de clientes afectados
  - Tiempo promedio muerto (min)
  - Tiempo promedio de resolución (min)

### Ejecución:
```bash
cd scripts
python train_gru.py
```
El modelo y los preprocesadores se guardarán en la carpeta `../models/`.

---

## Predicción

El archivo `scripts/predict.py` permite hacer una predicción ingresando:
- Fecha (YYYY-MM-DD)
- Causa de incidencia (texto)

### Ejecución:
```bash
cd scripts
python predict.py
```
Los resultados se mostrarán en consola.

---
Referencias:
https://github.com/jingwenshi-dev/Weather-Forecasting-by-GRU-Transformer
