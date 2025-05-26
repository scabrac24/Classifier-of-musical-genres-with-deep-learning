import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
import joblib



# 1. Carga y preparación de datos
train_df = pd.read_csv("Data/train_data_final.csv")
test_df = pd.read_csv("Data/test_data_final.csv")

# Mezclar los datos para evitar sesgos
train_df = train_df.sample(frac=1, random_state=42)

# Separar características y etiquetas
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Codificar etiquetas
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encoding
num_classes = len(np.unique(y_train_encoded))
y_train_cat = to_categorical(y_train_encoded, num_classes)
y_test_cat = to_categorical(y_test_encoded, num_classes)

# 2. Balanceo de clases 
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)
class_weights = dict(enumerate(class_weights))

# 3. División entrenamiento/validación
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_scaled, y_train_cat, 
    test_size=0.2, 
    random_state=42,
    stratify=y_train_cat  # Mantener proporción de clases
)

# 4. Modelo ¿
def build_model():
    model = Sequential([
        Dense(128, input_shape=(X_train_scaled.shape[1],), 
              activation='relu', 
              kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu', 
              kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model

model = build_model()
model.summary()

# 5. Callbacks
def lr_scheduler(epoch, lr):
    if epoch > 20:
        return lr * 0.1
    return lr

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 6. Entrenamiento
history = model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop, LearningRateScheduler(lr_scheduler)],
    verbose=1
)

# 7. Evaluación
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
print(f"\nResultados en Test:")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")

# 8. Gráficas
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
model.save('modelo_musica.h5')
