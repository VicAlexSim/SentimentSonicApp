import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from model import build_cnn_model

FEATURES_DIR = '../data/features/'
MODEL_SAVE_PATH = '../results/models/emotion_mfcc_model.h5'

# data
mfcc = np.load(os.path.join(FEATURES_DIR, 'mfccs.npy'))
labels = np.load(os.path.join(FEATURES_DIR, 'feature_labels.npy'))

print("Original MFCC shape:", mfcc.shape)
unique, counts = np.unique(labels, return_counts=True)
print("Label distribution:")
for label, count in zip(unique, counts):
    print(f"  {label}: {count}")

# reshape
if mfcc.shape[1] == 13:
    mfcc = np.transpose(mfcc, (0, 2, 1))
mfcc = mfcc[..., np.newaxis]
mfcc = mfcc.astype(np.float32)
mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-7)
print("Final MFCC shape:", mfcc.shape)

# labels
le = LabelEncoder()
y_int = le.fit_transform(labels)
y = to_categorical(y_int)
print("Class order:", le.classes_)

# class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_int), y=y_int)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print("🧮 Class weights:", class_weights_dict)

# split
X_train, X_val, y_train, y_val = train_test_split(
    mfcc, y, test_size=0.2, stratify=y_int, random_state=42
)

# build CNN
model = build_cnn_model(input_shape=mfcc.shape[1:], num_classes=y.shape[1])

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

print(f"Model trained and saved to {MODEL_SAVE_PATH}")
