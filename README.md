# Clasificador de Género Musical con Deep Learning

Este proyecto permite clasificar géneros musicales a partir de clips de audio de 3 segundos en formato MP3. Utiliza un modelo de red neuronal profunda entrenado con características extraídas del audio mediante `librosa`, y ofrece una interfaz web construida con Flask para que cualquier usuario pueda cargar un archivo y obtener una predicción.

---

## Características

- Carga de archivos MP3 desde una interfaz web sencilla.
- Recorte automático de los primeros 3 segundos de audio, evitando silencios iniciales para mejorar la precisión.
- Extracción de características acústicas con `librosa`.
- Clasificación del género con un modelo de aprendizaje profundo entrenado con TensorFlow/Keras.
- Visualización de espectrogramas.
- Mostrar el género predicho junto con el nivel de confianza.

---

## Requisitos

Asegúrate de tener Python 3.8 o superior.

Instala las dependencias ejecutando:

pip install -r requirements.txt

---

## Detalles del Modelo

El modelo fue entrenado usando características acústicas extraídas de clips de audio de 3 segundos. Las características incluyen:

40 coeficientes MFCC

Chroma STFT

Tonnetz

Zero Crossing Rate

Spectral Centroid

Spectral Rolloff

# Red Neuronal utilizada:

- Capa densa de 128 unidades con ReLU, regularización L2 y BatchNormalization

- Dropout de 0.4

- Capa densa de 64 unidades con ReLU, regularización L2 y BatchNormalization

- Dropout de 0.3

- Capa densa de 32 unidades con ReLU

- Capa de salida Softmax para clasificación multiclase

# Entrenamiento:

- Optimizador: Adam con learning rate = 0.0005

- Pérdida: Categorical Crossentropy

- Métricas: Accuracy, Precision, Recall

- Balanceo de clases con class_weight

- Early stopping con restauración de los mejores pesos

- Reducción de tasa de aprendizaje después de 20 épocas

# Evaluación en conjunto de prueba:

Se imprimen Accuracy, Precision y Recall sobre los datos de prueba al correr model.py


## Funcionamiento de la Aplicación Web

1) El usuario accede a la interfaz web en http://127.0.0.1:5000

2) Se permite cargar un archivo MP3 de duración mayor a 3 segundos.

3) El sistema recorta el audio evitando los primeros segundos (para evitar silencios).

4) Se extraen características y se normalizan usando el scaler.pkl entrenado.

5) El modelo predice el género y se genera un espectrograma del audio cargado.

6) El resultado mostrado incluye:

    -  Género musical

    -  Confianza del modelo

    -  Imagen del espectrograma generado