## Template code for now 
## For loading datasets, segmenting audio files, and preprocessing them for emotion recognition

import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Define constants (consider moving to a config file later)
RAW_DATA_DIR = '../data/raw/'
PROCESSED_DATA_DIR = '../data/processed/'
SAMPLE_RATE = 16000 # Standard sample rate
SEGMENT_LENGTH_SEC = 3 # Example segment length

def find_audio_files(data_dir):
    """
    Loads audio paths and labels from IEMOCAP metadata and from CREMA-D filenames.
    Returns two lists: file paths and emotion labels.
    """
    audio_files = []
    labels = []

    # 1. IEMOCAP (assumes CSV is in ../data/metadata/)
    iemocap_metadata_path = os.path.join('../data/metadata/iemocap_metadata.csv')
    if os.path.exists(iemocap_metadata_path):
        df = pd.read_csv(iemocap_metadata_path)
        target_emotions = ['fru', 'ang', 'neu']
        df = df[df['emotion'].isin(target_emotions)]
        df['full_path'] = df['path'].apply(lambda p: os.path.join(data_dir, p))
        audio_files += df['full_path'].tolist()
        labels += df['emotion'].tolist()
        print(f"IEMOCAP: Loaded {len(df)} labeled files.")

    # 2. CREMA-D (parse from filenames)
    crema_dir = os.path.join(data_dir, 'crema-d')
    if os.path.exists(crema_dir):
        for file in os.listdir(crema_dir):
            if file.endswith('.wav'):
                label = parse_label_from_filepath(file)
                if label in ['anger', 'neutral']:  # Only include relevant ones
                    audio_files.append(os.path.join(crema_dir, file))
                    labels.append(label)
        print(f"CREMA-D: Loaded {len(audio_files)} files after filtering.")

    return audio_files, labels

def parse_label_from_filepath(filename):
    """
    Extract emotion code from CREMA-D filename, e.g., '1001_IEO_ANG_XX.wav' → 'ANG' → 'anger'
    """
    parts = filename.split('_')
    if len(parts) > 2:
        emotion_code = parts[2]
        emotion_map = {
            'ANG': 'anger',
            'HAP': 'happy',
            'SAD': 'sad',
            'NEU': 'neutral',
            'DIS': 'disgust',
            'FEA': 'fear'
        }
        return emotion_map.get(emotion_code, 'unknown')
    return 'unknown'


def load_and_segment_audio(filepath, sr=SAMPLE_RATE, segment_len_sec=SEGMENT_LENGTH_SEC):
    """
    Loads an audio file and segments it into fixed-length chunks.
    Handles potential errors during loading.
    Applies basic preprocessing like resampling.
    """
    segments = []
    try:
        # Load audio file
        audio, current_sr = librosa.load(filepath, sr=None, mono=True) # Load original SR

        # Resample if necessary
        if current_sr != sr:
            audio = librosa.resample(audio, orig_sr=current_sr, target_sr=sr)

        # Calculate segment length in samples
        segment_len_samples = int(segment_len_sec * sr)

        # Pad audio if it's shorter than segment length
        if len(audio) < segment_len_samples:
             padding = segment_len_samples - len(audio)
             audio = np.pad(audio, (0, padding), 'constant')

        # Create segments
        num_segments = int(np.ceil(len(audio) / segment_len_samples))
        for i in range(num_segments):
            start = i * segment_len_samples
            end = start + segment_len_samples
            segment = audio[start:end]

            # Ensure segment is exactly the required length (handle last segment)
            if len(segment) < segment_len_samples:
                 padding = segment_len_samples - len(segment)
                 segment = np.pad(segment, (0, padding), 'constant')

            segments.append(segment)

    except Exception as e:
        print(f"Error loading or processing {filepath}: {e}")
        return [] # Return empty list on error

    return segments

def preprocess_audio_segment(segment, sr=SAMPLE_RATE):
    """
    Applies preprocessing steps like noise reduction (optional) and normalization.
    Placeholder for now.
    """
    # 1. Noise Reduction (Optional - requires specific libraries/techniques)
    # Example: spectral gating, Wiener filter (can be complex)
    # segment_denoised = apply_noise_reduction(segment, sr)

    # 2. Normalization (e.g., peak normalization)
    peak_val = np.max(np.abs(segment))
    if peak_val > 0:
        segment_normalized = segment / peak_val
    else:
        segment_normalized = segment # Avoid division by zero

    return segment_normalized # or segment_denoised if implemented

def process_dataset(raw_data_dir, processed_data_dir, sr=SAMPLE_RATE, segment_len_sec=SEGMENT_LENGTH_SEC):
    """
    Main function to find audio files, load, segment, preprocess,
    and save the processed data.
    """
    print("Starting dataset processing...")
    audio_files, labels = find_audio_files(raw_data_dir)

    if not audio_files:
        print("No audio files found. Exiting.")
        return

    all_segments = []
    all_labels = []
    file_origins = [] # Keep track of which file segment came from

    for filepath, label in zip(audio_files, labels):
        print(f"Processing: {filepath} with label: {label}")
        segments = load_and_segment_audio(filepath, sr, segment_len_sec)
        for segment in segments:
            processed_segment = preprocess_audio_segment(segment, sr)
            all_segments.append(processed_segment)
            all_labels.append(label)
            file_origins.append(os.path.basename(filepath))

    if not all_segments:
        print("No segments were processed successfully.")
        return

    # Convert to NumPy arrays
    all_segments = np.array(all_segments)
    all_labels = np.array(all_labels)
    file_origins = np.array(file_origins)

    # Ensure processed data directory exists
    os.makedirs(processed_data_dir, exist_ok=True)

    # Save processed data (consider saving as .npy or in a structured format)
    # Example: Save segments and labels separately
    segments_path = os.path.join(processed_data_dir, 'segments.npy')
    labels_path = os.path.join(processed_data_dir, 'labels.npy')
    origins_path = os.path.join(processed_data_dir, 'origins.npy')

    np.save(segments_path, all_segments)
    np.save(labels_path, all_labels)
    np.save(origins_path, file_origins)

    print(f"Processed {len(all_segments)} segments.")
    print(f"Saved segments to: {segments_path}")
    print(f"Saved labels to: {labels_path}")
    print(f"Saved origins to: {origins_path}")

    # Optional: Create train/validation/test splits here or later
    # X_train, X_test, y_train, y_test = train_test_split(all_segments, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    # print("Data split into train/test sets.")
    # Save splits if needed

if __name__ == "__main__":
    # Example usage:
    # Make sure the paths are correct relative to where you run the script from
    # If running from SentimentSonic_Project/src/, use '../data/raw/' etc.
    # If running from SentimentSonic_Project/, use 'data/raw/' etc.
    # Adjust RAW_DATA_DIR and PROCESSED_DATA_DIR accordingly or use absolute paths.

    # Assuming script is run from the 'src' directory:
    process_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)