## building CNN, LSTM, and CNN-LSTM models for emotion recognition

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    LSTM, Bidirectional, TimeDistributed, BatchNormalization, Activation
)

def build_mfcc_branch(input_tensor):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    return x

def build_spec_branch(input_tensor):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    return x

def build_cnn_model(input_shape, num_classes):

    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
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
        LSTM(64, return_sequences=True), # return sequences for stacking LSTMs
        Dropout(0.3),
        LSTM(64, return_sequences=False), # only last output for classification
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

    # CNN part to extract spatial features
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

    shape = tf.keras.backend.int_shape(x)
    x = tf.keras.layers.Reshape((shape[2], shape[1] * shape[3]))(x)

    # LSTM
    x = Bidirectional(LSTM(64, return_sequences=False))(x) # use bidirectional for context
    x = Dropout(0.3)(x)

    # final classification layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_Model")

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # instantiation
    num_emotion_classes = 5 # frustration, anger, neutral, happy, sad

    # spectrogram-based CNN/CNN-LSTM
    spectrogram_height = 128     # mel bands
    spectrogram_width = 400      # time steps
    spectrogram_channels = 1
    cnn_input_shape = (spectrogram_height, spectrogram_width, spectrogram_channels)

    # MFCC-based LSTM
    mfcc_timesteps = 400         # time steps
    mfcc_features = 13           # MFCC coefficients
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