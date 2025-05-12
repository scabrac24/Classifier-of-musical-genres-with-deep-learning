#  Music Genre Classification from Audio Features

Este proyecto implementa un modelo de clasificación multiclase para identificar géneros musicales basándose en características extraídas de clips de audio de 3 segundos. Utiliza datos derivados de espectrogramas, MFCCs, cromas, tonnetz y otras propiedades espectrales, entrenado con TensorFlow.

---

## Dataset

El dataset contiene 16 géneros musicales, con 2500 clips por género. Cada clip de audio fue extraído de mezclas de YouTube usando Ableton y procesado para obtener estadísticas como media y desviación estándar de las siguientes características:

- **RMSE**
- **Zero Crossing Rate**
- **Spectral Features**: centroid, bandwidth, rolloff
- **MFCCs**: 40 coeficientes (`mfcc1` - `mfcc40`)
- **Chroma Features**: `chroma1` - `chroma12`
- **Tonnetz Features**: `tonnetz1` - `tonnetz6`
- **Chroma CQT**, **Spectral Contrast**
- **Label**: género musical asociado a cada clip

---

## Modelo

El modelo está construido usando `TensorFlow` y `Keras`. Es una red neuronal completamente conectada (*fully connected feedforward*) con la siguiente arquitectura:

Input: 131 features
↓
Dense(256, ReLU) + Dropout(0.3)
↓
Dense(128, ReLU) + Dropout(0.3)
↓
Dense(64, ReLU)
↓
Output: Dense(16, Softmax)


---

## Entrenamiento

- **Entradas**: Características numéricas normalizadas (`StandardScaler`)
- **Etiquetas**: Géneros codificados con `LabelEncoder` y convertidos a one-hot
- **Función de pérdida**: `categorical_crossentropy`
- **Optimizador**: `Adam`
- **Épocas**: 30
- **Tamaño de batch**: 64
- **Validación**: 80% entrenamiento / 20% validación

---

## Evaluación

La precisión en el conjunto de prueba (`test_data_final.csv`) se reporta al final del entrenamiento. También se puede visualizar la evolución de la precisión y la pérdida por época usando `matplotlib`.

---


