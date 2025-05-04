# SentimentSonic: Audio Emotion Detection (CS 4375.501 - Option 3)

## Group Members

*   Victor Sim (vas230001@utdallas.edu)
*   Arhum Khan (axk210013@utdallas.edu)
*   Vy Nguyen (vpn200000@utdallas.edu)
*   Marco Frescas (mdf200001@utdallas.edu)
*   Brody Roche (bar210002@utdallas.edu)

## Project Overview

This project aims to develop a real-time audio analysis system ("SentimentSonic") capable of detecting customer frustration or negative emotional states in call center audio streams. The goal is to trigger timely intervention alerts, enabling supervisors or agents to proactively address issues, thereby improving customer satisfaction and reducing escalations.

## 1. Problem Statement

*   **Goal:** Monitor call center audio in real-time to detect customer frustration and trigger intervention alerts.
*   **Context:** Undetected customer frustration in call centers leads to poor service outcomes and negative brand perception. Real-time detection allows for immediate intervention and de-escalation.
*   **Key Challenges:** Handling audio variability (noise, quality, accents), achieving real-time processing speeds, and specifically identifying frustration among other emotions.
*   **Impact:** Potential to reduce call durations, improve customer satisfaction, lower escalation rates, and provide agent feedback.

## 2. Dataset

This project will primarily utilize public audio emotion datasets. Initial candidates include:

*   **IEMOCAP:** Multimodal dataset with scripted/improvised scenarios and detailed emotion annotations (including frustration). High quality, but may require adaptation from acted scenarios to real call center context.
*   **CREMA-D:** Audio-visual dataset of actors expressing emotions (anger, disgust, fear, happiness, neutral). Provides diverse vocal expressions but may differ from spontaneous customer speech.

**Data Preprocessing Steps:**
*   Audio Segmentation (e.g., sentence or turn-level)
*   Noise Reduction and Normalization
*   Data Augmentation (e.g., adding noise, pitch/speed variations)

*Note: Datasets need to be downloaded separately and placed in the `data/raw/` directory.*

## 3. ML Techniques

*   **Feature Extraction:**
    *   MFCCs (Mel-Frequency Cepstral Coefficients)
    *   Spectrograms
*   **Model Architectures:**
    *   Convolutional Neural Networks (CNNs) for spatial patterns in spectrograms.
    *   Recurrent Neural Networks (RNNs), specifically LSTMs, for temporal dependencies in audio sequences.
    *   Hybrid CNN-LSTM models combining spatial and temporal feature learning.
*   **Transfer Learning:** Fine-tuning pre-trained audio models on target datasets may be explored.

## Project Structure
SentimentSonic_Project/
├── README.md                 # Project overview, setup, and usage instructions
├── requirements.txt          # Python libraries needed
├── data/
│   ├── raw/                  # Place for downloaded raw datasets (e.g., IEMOCAP, CREMA-D) - Add .gitignore here later
│   └── processed/            # Place for cleaned, segmented, feature-extracted data - Add .gitignore here later
├── notebooks/
│   ├── 01_data_exploration.ipynb  # For initial dataset analysis and visualization
│   ├── 02_feature_extraction.ipynb # For experimenting with MFCCs, Spectrograms
│   └── 03_model_prototyping.ipynb # For building and testing initial models
├── src/
│   ├── __init__.py           # Makes src a Python package
│   ├── data_loader.py        # Scripts for loading and preprocessing data
│   ├── feature_extractor.py  # Scripts for extracting audio features
│   ├── model.py              # Defines the CNN, LSTM, or Hybrid model architectures
│   ├── train.py              # Script to train the model
│   ├── evaluate.py           # Script to evaluate the model performance
│   └── utils.py              # Utility functions (e.g., saving/loading models, plotting)
└── results/
    ├── models/               # Saved trained model files
    └── plots/                # Generated plots (e.g., confusion matrix, loss curves)



## Setup
 
1. Clone the repository.
2. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4. Download datasets and place them correctly.
    * Create the `data/raw/` directory if it doesn't already exist.
    * CREMA-D
        * Download the audio `.wav` files from [Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad) or your source of choice.
        * Place the `.wav` files directly inside `data/raw/crema-d/`.
    * IEMOCAP (optional for now)
        * Apply for access via https://sail.usc.edu/iemocap/.
        * Once approved, download and extract the files, and place the `.wav` files directly inside `data/raw/iemocap/`.
        * Additionally, download this `.csv` file from [Kaggle](https://www.kaggle.com/datasets/samuelsamsudinng/iemocap-emotion-speech-database) and place it directly in `data/metadata/`. 

## Usage

1. **Data Preprocessing:**
    ```bash
    cd src
    python data_loader.py
    ```
2. **Extract Audio Features (MFCCs + Spectrograms)**
    ```bash
    python extract_features.py
    ```
3. **Train the Model (choose one):**
    ```bash
    python train_mfcc_only.py       # MFCC-only model
    python train_spec_only.py       # Spectrogram-only model
    python train_dual_input.py      # Combined model (best so far, USE THIS ONE)
    ```
4. **Evaluate Model Performance:**
    ```bash
    python evaluate.py
    ```
5. **Run the Web App** (upload a `.wav` file to test):
    ```bash
    python app.py
    ```
