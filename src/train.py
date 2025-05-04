import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout, Flatten
from model import build_mfcc_branch, build_spec_branch

FEATURES_DIR = '../data/features/'
MODEL_SAVE_PATH = '../results/models/emotion_dual_model.h5'

# load features
mfcc = np.load(os.path.join(FEATURES_DIR, 'mfccs.npy'))
spec = np.load(os.path.join(FEATURES_DIR, 'spectrograms.npy'))
labels = np.load(os.path.join(FEATURES_DIR, 'feature_labels.npy'))

# 🔍 Check label distribution
unique, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique, counts):
    print(f"{label}: {count}")

# Transpose and reshape MFCC
if mfcc.shape[1] == 13:
    mfcc = np.transpose(mfcc, (0, 2, 1))  # (samples, 400, 13)
mfcc = mfcc[..., np.newaxis]

# Transpose and reshape Spectrogram
if spec.shape[1] == 128:
    spec = np.transpose(spec, (0, 2, 1))  # (samples, time, 128)
spec = spec[..., np.newaxis]


# Normalize MFCC
mfcc_mean = np.mean(mfcc)
mfcc_std = np.std(mfcc)
mfcc = (mfcc - mfcc_mean) / mfcc_std

# Normalize Spectrogram
spec_mean = np.mean(spec)
spec_std = np.std(spec)
spec = (spec - spec_mean) / spec_std


# Encode labels
le = LabelEncoder()
y_int = le.fit_transform(labels)
y = to_categorical(y_int)
print("Class label order:", le.classes_)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_int), y=y_int)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Split
from sklearn.model_selection import train_test_split
mfcc_train, mfcc_val, spec_train, spec_val, y_train, y_val = train_test_split(
    mfcc, spec, y, test_size=0.2, stratify=y_int, random_state=42
)

# Build dual-input model
mfcc_input = Input(shape=mfcc.shape[1:])
spec_input = Input(shape=spec.shape[1:])

mfcc_branch = build_mfcc_branch(mfcc_input)
spec_branch = build_spec_branch(spec_input)

combined = Concatenate()([mfcc_branch, spec_branch])
x = Flatten()(combined)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(y.shape[1], activation='softmax')(x)

model = Model(inputs=[mfcc_input, spec_input], outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
history = model.fit(
    [mfcc_train, spec_train], y_train,
    validation_data=([mfcc_val, spec_val], y_val),
    epochs=20,
    batch_size=32,
    callbacks=[checkpoint],
    verbose=1,
    class_weight=class_weights_dict
)

print(f"Model trained and saved to {MODEL_SAVE_PATH}")