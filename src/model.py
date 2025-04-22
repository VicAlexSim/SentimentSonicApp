## Template code for now
## For building CNN, LSTM, and CNN-LSTM models for emotion recognition from audio features

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    LSTM, Bidirectional, TimeDistributed, BatchNormalization, Activation
)

def build_cnn_model(input_shape, num_classes):
    """
    Builds a basic CNN model for spectrogram input.
    input_shape: (height, width, channels) e.g., (128, 128, 1) for spectrograms
    num_classes: Number of output emotion classes
    """
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') # Use 'softmax' for multi-class classification
    ], name="CNN_Model")

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', # Use if labels are one-hot encoded
                  # loss='sparse_categorical_crossentropy', # Use if labels are integers
                  metrics=['accuracy'])
    return model

def build_lstm_model(input_shape, num_classes):
    """
    Builds a basic LSTM model for sequence input (e.g., MFCCs over time).
    input_shape: (timesteps, features) e.g., (None, 13) for MFCCs
    num_classes: Number of output emotion classes
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True), # Return sequences for stacking LSTMs
        Dropout(0.3),
        LSTM(64, return_sequences=False), # Only last output for classification
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name="LSTM_Model")

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_cnn_lstm_model(input_shape, num_classes):
    """
    Builds a hybrid CNN-LSTM model. Assumes input is spectrogram-like.
    input_shape: (height, width, channels) e.g., (128, 128, 1)
    num_classes: Number of output emotion classes
    """
    inputs = Input(shape=input_shape)

    # CNN part to extract spatial features (treat time axis of spectrogram as width)
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Reshape features to be sequence for LSTM
    # Shape after CNN might be (batch, height', width', channels')
    # We need (batch, timesteps, features) for LSTM
    # Example: Reshape based on pooling layers applied
    # This reshape depends heavily on the CNN architecture and input shape
    # Assuming width dimension represents time after convolutions
    current_shape = tf.keras.backend.int_shape(x) # (None, height', width', channels')
    # Combine height' and channels' into a single feature dimension
    target_shape = (-1, current_shape[2], current_shape[1] * current_shape[3])
    x = tf.keras.layers.Reshape(target_shape)(x) # (batch, width', height'*channels')

    # LSTM part to process temporal sequence
    x = Bidirectional(LSTM(64, return_sequences=False))(x) # Use Bidirectional for context
    x = Dropout(0.3)(x)

    # Final classification layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_Model")

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # Example instantiation (replace with actual shapes and num_classes)
    num_emotion_classes = 5 # Example: frustration, anger, neutral, happy, sad

    # Example Spectrogram shape (adjust based on your feature extraction)
    spectrogram_height = 128
    spectrogram_width = 128 # This represents time steps usually
    spectrogram_channels = 1
    cnn_input_shape = (spectrogram_height, spectrogram_width, spectrogram_channels)

    # Example MFCC shape (adjust based on your feature extraction)
    mfcc_timesteps = 128 # Example fixed length after padding/truncating
    mfcc_features = 13
    lstm_input_shape = (mfcc_timesteps, mfcc_features)

    print("Building CNN Model...")
    cnn_model = build_cnn_model(cnn_input_shape, num_emotion_classes)
    cnn_model.summary()

    print("\nBuilding LSTM Model...")
    lstm_model = build_lstm_model(lstm_input_shape, num_emotion_classes)
    lstm_model.summary()

    print("\nBuilding CNN-LSTM Model...")
    cnn_lstm_model = build_cnn_lstm_model(cnn_input_shape, num_emotion_classes)
    cnn_lstm_model.summary()