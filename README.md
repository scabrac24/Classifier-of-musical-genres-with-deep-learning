# Clasificador de Género Musical con Deep Learning

Este proyecto permite clasificar géneros musicales a partir de clips de audio de 3 segundos en formato MP3. Utiliza un modelo de red neuronal profunda entrenado con características extraídas del audio mediante `librosa`, y ofrece una interfaz web construida con Flask para que cualquier usuario pueda cargar un archivo y obtener una predicción. Para entrenar el modelo se uso el siguiente Dataset extraido de la plataforma Kaggle : https://www.kaggle.com/datasets/sivadithiyan/edm-music-genres , autoria de Sivadithiyan



## Características

- Carga de archivos MP3 desde una interfaz web sencilla.
- Recorte automático de un fragmento de 3 segundos del archivo MP3 ingresado (este recorte se aplica a los 3 segundos siguientes despues de la espera que se puede modificar en el codigo), evitando silencios iniciales para mejorar la precisión.
- Extracción de características acústicas con `librosa`.
- Clasificación del género con un modelo de aprendizaje profundo entrenado con TensorFlow/Keras.
- Visualización de espectrogramas.
- Mostrar el género predicho junto con el nivel de confianza.
  
### Los generos que el modelo es capaz de reconocer son:

- Ambient 🌌
- Big Room House 🏠
- Drum and Bass 🥁
- Dubstep 🎵
- Future Garage/Wave Trap 🌊
- Hardcore 🔊
- Hardstyle 💥
- House 🏡
- Lo-fi 🎶
- Moombahton/Reggaeton 🎵🌴
- Phonk 🔥
- Psytrance 🌀
- Synthwave 🎹
- Techno 🎛️
- Trance 🚀
- Trap ⛓️



## Requisitos

Asegurarse de tener Python 3.8 o superior.

Instalar las dependencias ejecutando:

pip install -r requirements.txt


---

# Detalles del Modelo

El modelo fue entrenado usando características acústicas extraídas de clips de audio de 3 segundos. Las características incluyen:

40 coeficientes MFCC

Chroma STFT

Tonnetz

Zero Crossing Rate

Spectral Centroid

Spectral Rolloff



## Red Neuronal utilizada:


- Capa densa de 128 unidades con ReLU, regularización L2 y BatchNormalization

- Dropout de 0.4

- Capa densa de 64 unidades con ReLU, regularización L2 y BatchNormalization

- Dropout de 0.3

- Capa densa de 32 unidades con ReLU

- Capa de salida Softmax para clasificación multiclase



## Entrenamiento:

- Optimizador: Adam con learning rate = 0.0005

- Pérdida: Categorical Crossentropy

- Métricas: Accuracy, Precision, Recall

- Balanceo de clases con class_weight

- Early stopping con restauración de los mejores pesos

- Reducción de tasa de aprendizaje después de 20 épocas



## Evaluación en conjunto de prueba:

Se imprimen Accuracy, Precision y Recall sobre los datos de prueba al correr model.py

---

# Funcionamiento de la Aplicación Web

1) Acceder a la carpeta Model con cd .\Model\

2) Ejecutar python app.py en la terminal

3) Acceder a la interfaz web en http://127.0.0.1:5000

4) Cargar un archivo MP3 de duración mayor a 3 segundos.

5) El sistema recorta el audio evitando la espera ingresada en el codigo (para evitar silencios).

6) Se extraen características y se normalizan usando el scaler.pkl entrenado.

7) El modelo predice el género y se genera un espectrograma del audio cargado.

8) El resultado mostrado incluye:

    -  Género musical

    -  Confianza del modelo

    -  Imagen del espectrograma generado
