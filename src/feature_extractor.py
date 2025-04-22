## Template code for now 
## For feature extraction of audio files that are already segmented and preprocessed

import librosa
import numpy as np
import os

# Define constants (consider moving to a config file later)
SAMPLE_RATE = 16000
N_MFCC = 13       # Number of MFCCs to return
N_FFT = 2048      # Window size for FFT
HOP_LENGTH = 512  # Hop length for FFT
N_MELS = 128      # Number of Mel bands for spectrogram

PROCESSED_DATA_DIR = '../data/processed/'
FEATURES_DIR = '../data/features/' # New directory for features

def extract_mfcc(audio_segment, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Extracts MFCCs from an audio segment.
    """
    try:
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        # Optionally add delta features
        # delta_mfccs = librosa.feature.delta(mfccs)
        # delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        # mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
        return mfccs # or mfccs_features
    except Exception as e:
        print(f"Error extracting MFCCs: {e}")
        return None

def extract_spectrogram(audio_segment, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Extracts a Mel Spectrogram from an audio segment.
    Converts power spectrogram to dB scale.
    """
    try:
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        # Convert to decibels (log scale)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return log_mel_spectrogram
    except Exception as e:
        print(f"Error extracting Spectrogram: {e}")
        return None

def process_features(processed_data_dir, features_dir):
    """
    Loads processed segments and extracts features (MFCCs, Spectrograms).
    Saves the features.
    """
    print("Starting feature extraction...")
    segments_path = os.path.join(processed_data_dir, 'segments.npy')
    labels_path = os.path.join(processed_data_dir, 'labels.npy') # Keep labels aligned

    if not os.path.exists(segments_path) or not os.path.exists(labels_path):
        print(f"Processed data not found at {processed_data_dir}. Run data_loader.py first.")
        return

    segments = np.load(segments_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    all_mfccs = []
    all_spectrograms = []
    valid_labels = []

    for i, segment in enumerate(segments):
        print(f"Extracting features for segment {i+1}/{len(segments)}")
        mfcc = extract_mfcc(segment)
        spectrogram = extract_spectrogram(segment)

        if mfcc is not None and spectrogram is not None:
            all_mfccs.append(mfcc)
            all_spectrograms.append(spectrogram)
            valid_labels.append(labels[i]) # Keep label for valid features
        else:
            print(f"Skipping segment {i+1} due to feature extraction error.")

    if not all_mfccs or not all_spectrograms:
        print("No features were extracted successfully.")
        return

    # Ensure features directory exists
    os.makedirs(features_dir, exist_ok=True)

    # Save features (consider padding/truncating features to have consistent dimensions if needed for models)
    # Note: Saving lists of arrays might require different handling (e.g., saving each as separate file or padding)
    # For simplicity, saving as .npy assuming consistent shapes or handling downstream
    mfccs_path = os.path.join(features_dir, 'mfccs.npy')
    spectrograms_path = os.path.join(features_dir, 'spectrograms.npy')
    feature_labels_path = os.path.join(features_dir, 'feature_labels.npy') # Labels corresponding to features

    # Need to ensure features have consistent shapes before saving as single array
    # This might involve padding or choosing a fixed length
    # Placeholder: Save as object array, requires careful loading later
    np.save(mfccs_path, np.array(all_mfccs, dtype=object), allow_pickle=True)
    np.save(spectrograms_path, np.array(all_spectrograms, dtype=object), allow_pickle=True)
    np.save(feature_labels_path, np.array(valid_labels))

    print(f"Extracted features for {len(valid_labels)} segments.")
    print(f"Saved MFCCs to: {mfccs_path}")
    print(f"Saved Spectrograms to: {spectrograms_path}")
    print(f"Saved corresponding labels to: {feature_labels_path}")


if __name__ == "__main__":
    # Assuming script is run from the 'src' directory:
    process_features(PROCESSED_DATA_DIR, FEATURES_DIR)