import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


"""
1 - Preparación de datos obtenidos de Kaggle - EDM Music Genres by Sivadithiyan official
"""

# Cargar datos
train_df = pd.read_csv("Data/train_data_final.csv")
test_df = pd.read_csv("Data/test_data_final.csv")

print(train_df.columns.tolist())

# Separar características y etiquetas
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Codificar etiquetas de texto a números
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


"""
2 - Creación del modelo de red neuronal
"""

# One-hot encoding de las etiquetas
num_classes = len(np.unique(y_train_encoded))
y_train_cat = to_categorical(y_train_encoded, num_classes)
y_test_cat = to_categorical(y_test_encoded, num_classes)

# Modelo
model = Sequential([
    Dense(256, input_shape=(X_train_scaled.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

"""
3 - Entrenamiento del modelo
"""

history = model.fit(
    X_train_scaled, y_train_cat,
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    verbose=1
)


test_loss, test_acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
print(f"Accuracy en test: {test_acc:.4f}")


plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión durante el entrenamiento')
plt.show()
