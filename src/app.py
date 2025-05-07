import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import uuid

UPLOAD_FOLDER = '../uploads/'
MODEL_PATH = '../results/models/emotion_dual_model.h5'

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model(MODEL_PATH)

# Label map (update if needed)
LABELS = ['anger', 'happy', 'neutral', 'sad', 'frustration']

def extract_features(file_path):
    # Load audio and extract MFCCs and Spectrogram
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    
    # Pad or truncate
    mfcc = librosa.util.fix_length(mfcc, size=400, axis=1)
    spec = librosa.util.fix_length(spec, size=400, axis=1)

    # Reshape for model input
    mfcc = np.transpose(mfcc, (1, 0))[np.newaxis, ..., np.newaxis]
    spec = np.transpose(spec, (1, 0))[np.newaxis, ..., np.newaxis]

    # Normalize (same as training)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-7)
    spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-7)
    
    return mfcc, spec

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['audio_file']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.wav")
    file.save(filename)

    mfcc, spec = extract_features(filename)
    prediction = model.predict([mfcc, spec])
    predicted_label = LABELS[np.argmax(prediction)]

    os.remove(filename)  # clean up after prediction
    return render_template('index.html', prediction=predicted_label)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
