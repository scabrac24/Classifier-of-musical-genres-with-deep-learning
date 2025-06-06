from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import uuid
import soundfile as sf 



# Define rutas absolutas para templates y static
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


UPLOAD_FOLDER = 'uploads'
SPECTROGRAM_FOLDER = 'static/spectrograms'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)

# Cargar modelo y scaler
model = tf.keras.models.load_model("modelo_musica.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


def crop_audio_to(file_path, skip_start_sec=80.0):
    y, sr = librosa.load(file_path, sr=None)
    total_samples = len(y)
    total_duration = librosa.get_duration(y=y, sr=sr)

    # Si el audio es muy corto, tomar los últimos 3 segundos o lo que haya
    if total_duration <= 3.0:
        return  # No hace falta recortar

    start_sample = int(skip_start_sec * sr)
    end_sample = start_sample + int(3 * sr)

    # Si el final se pasa del total, recortar desde el final
    if end_sample > total_samples:
        end_sample = total_samples
        start_sample = max(0, end_sample - int(3 * sr))

    y_crop = y[start_sample:end_sample]
    sf.write(file_path, y_crop, sr)
 


# Generar espectrograma y guardarlo como imagen
def generate_spectrogram(file_path, output_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Extraer características
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3.0)

    features = {}
    rmse = librosa.feature.rms(y=y)[0]
    features['rmse_mean'] = np.mean(rmse)
    features['rmse_std'] = np.std(rmse)

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spec_cent)
    features['spectral_centroid_std'] = np.std(spec_cent)

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spec_bw)
    features['spectral_bandwidth_std'] = np.std(spec_bw)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_std'] = np.std(rolloff)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_std'] = np.std(zcr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    for i in range(40):
        features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
        features[f'mfcc{i+1}_std'] = np.std(mfcc[i])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features[f'chroma{i+1}_mean'] = np.mean(chroma[i])
        features[f'chroma{i+1}_std'] = np.std(chroma[i])

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    for i in range(6):
        features[f'tonnetz{i+1}_mean'] = np.mean(tonnetz[i])
        features[f'tonnetz{i+1}_std'] = np.std(tonnetz[i])

    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    features['chroma_cqt_mean'] = np.mean(chroma_cqt)
    features['chroma_cqt_std'] = np.std(chroma_cqt)

    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast_mean'] = np.mean(spec_contrast)
    features['spectral_contrast_std'] = np.std(spec_contrast)

    return pd.DataFrame([features])

# Ruta principal (HTML)
@app.route('/')
def index():
    return render_template('index.html')

# Ruta de predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró archivo'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        # Recortar el audio
        crop_audio_to(file_path)

        # Extraer características y predecir
        features = extract_features(file_path)
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]  # primera fila del batch

        max_prob_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([max_prob_index])[0]
        confidence = float(prediction[max_prob_index]) * 100
        error = 100 - confidence

        # Generar espectrograma
        spectrogram_filename = f"{uuid.uuid4().hex}.png"
        spectrogram_path = os.path.join(SPECTROGRAM_FOLDER, spectrogram_filename)
        generate_spectrogram(file_path, spectrogram_path)

        # Devolver JSON para el frontend
        return jsonify({
            'genre': predicted_label,
            'confidence': round(confidence, 2),
            'error': round(error, 2),
            'spectrogram': spectrogram_filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)




if __name__ == '__main__':
    app.run(debug=True)
