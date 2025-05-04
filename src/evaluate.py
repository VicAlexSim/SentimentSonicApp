# written to evaluate train.py (the combined model)

import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

FEATURES_DIR = '../data/features/'
MODEL_PATH = '../results/models/emotion_dual_model.h5'

# Load features
mfcc = np.load(os.path.join(FEATURES_DIR, 'mfccs.npy'))
spec = np.load(os.path.join(FEATURES_DIR, 'spectrograms.npy'))
labels = np.load(os.path.join(FEATURES_DIR, 'feature_labels.npy'))

# Transpose and reshape MFCC
if mfcc.shape[1] == 13:
    mfcc = np.transpose(mfcc, (0, 2, 1))
mfcc = mfcc[..., np.newaxis]

# Transpose and reshape Spectrogram
if spec.shape[1] == 128:
    spec = np.transpose(spec, (0, 2, 1))
spec = spec[..., np.newaxis]

# Normalize (same as in train.py)
mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
spec = (spec - np.mean(spec)) / np.std(spec)

# Encode labels
le = LabelEncoder()
y_int = le.fit_transform(labels)
y = to_categorical(y_int)

print("\n📊 Label distribution:")
for label, count in zip(*np.unique(labels, return_counts=True)):
    print(f"{label}: {count}")
print(f"\n✅ MFCC shape: {mfcc.shape}")
print(f"✅ Spec shape: {spec.shape}")
print("Class label order:", le.classes_)

# Split into test set
_, mfcc_test, _, spec_test, _, y_test, _, y_test_int = train_test_split(
    mfcc, spec, y, y_int, test_size=0.2, stratify=y_int, random_state=42
)

# Load model
model = load_model(MODEL_PATH)

# Evaluate
loss, accuracy = model.evaluate([mfcc_test, spec_test], y_test, verbose=1)
print(f"\n✅ Test Accuracy: {accuracy:.4f}")

# Predict and analyze
y_pred = model.predict([mfcc_test, spec_test])
y_pred_int = np.argmax(y_pred, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_test_int, y_pred_int, target_names=le.classes_))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test_int, y_pred_int))