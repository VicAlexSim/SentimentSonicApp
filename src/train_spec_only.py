import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from model import build_cnn_lstm_model

FEATURES_DIR = '../data/features/'
MODEL_SAVE_PATH = '../results/models/emotion_spec_model.h5'

# data
spec = np.load(os.path.join(FEATURES_DIR, 'spectrograms.npy'))
labels = np.load(os.path.join(FEATURES_DIR, 'feature_labels.npy'))

# reshape
if spec.shape[1] == 128:
    spec = np.transpose(spec, (0, 2, 1))  # (samples, time, features)
spec = spec[..., np.newaxis]  # (samples, time, features, 1)

# normalize
spec = spec / np.max(np.abs(spec))

# labels
le = LabelEncoder()
y_int = le.fit_transform(labels)
y = to_categorical(y_int)

# class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_int), y=y_int)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# training split
X_train, X_val, y_train, y_val = train_test_split(
    spec, y, test_size=0.2, stratify=y_int, random_state=42
)

# build CNN-LSTM model
input_shape = spec.shape[1:]
num_classes = y.shape[1]
model = build_cnn_lstm_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# save best model
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max')

# train
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[checkpoint],
    verbose=2,
    class_weight=class_weights_dict
)

print(f"✅ Model trained and saved to {MODEL_SAVE_PATH}")
