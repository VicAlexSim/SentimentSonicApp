## feature extraction of audio that was segmented and preprocessed

import librosa
import numpy as np
import os

# constants
SAMPLE_RATE = 16000
N_MFCC = 13       # num of MFCCs
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128      # num of mel bands for spectrogram

PROCESSED_DATA_DIR = '../data/processed/'
FEATURES_DIR = '../data/features/'
MAX_MFCC_TIME_STEPS = 400 
MAX_SPEC_TIME_STEPS = 400

def pad_or_truncate_feature(feature, max_len):
    """
    Pads or truncates a 2D feature array to (max_len, feature_dim)
    """
    current_len = feature.shape[1]
    if current_len > max_len:
        return feature[:, :max_len]
    elif current_len < max_len:
        pad_width = max_len - current_len
        return np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return feature


def extract_mfcc(audio_segment, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Extracts MFCCs from an audio segment.
    """
    try:
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return mfccs
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
    labels_path = os.path.join(processed_data_dir, 'labels.npy') # align labels

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
            # pad or truncate
            mfcc_padded = pad_or_truncate_feature(mfcc, MAX_MFCC_TIME_STEPS)
            spec_padded = pad_or_truncate_feature(spectrogram, MAX_SPEC_TIME_STEPS)

            all_mfccs.append(mfcc_padded)
            all_spectrograms.append(spec_padded)
            valid_labels.append(labels[i])
        else:
            print(f"Skipping segment {i+1} due to feature extraction error.")

    if not all_mfccs or not all_spectrograms:
        print("No features were extracted successfully.")
        return

    os.makedirs(features_dir, exist_ok=True)

    # save features
    mfccs_path = os.path.join(features_dir, 'mfccs.npy')
    spectrograms_path = os.path.join(features_dir, 'spectrograms.npy')
    feature_labels_path = os.path.join(features_dir, 'feature_labels.npy')

    # make sure features have consistent shapes before saving as array
    np.save(mfccs_path, np.array(all_mfccs), allow_pickle=False)
    np.save(spectrograms_path, np.array(all_spectrograms), allow_pickle=False)
    np.save(feature_labels_path, np.array(valid_labels))

    print(f"Extracted features for {len(valid_labels)} segments.")
    print(f"Saved MFCCs to: {mfccs_path}")
    print(f"Saved Spectrograms to: {spectrograms_path}")
    print(f"Saved corresponding labels to: {feature_labels_path}")


if __name__ == "__main__":
    process_features(PROCESSED_DATA_DIR, FEATURES_DIR)