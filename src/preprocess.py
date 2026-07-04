# src/preprocess.py

import librosa
import numpy as np
import soundfile as sf
import os

def load_audio(filepath, sample_rate=16000):
    """Load an audio file and return signal and sample rate"""
    y, sr = librosa.load(filepath, sr=sample_rate)
    print(f"✅ Loaded: {filepath}")
    print(f"   Duration: {len(y)/sr:.2f} seconds")
    print(f"   Sample Rate: {sr} Hz")
    return y, sr


def add_noise(clean_signal, noise_level=0.3):
    """Add white gaussian noise to a clean signal"""
    noise = np.random.normal(0, 1, len(clean_signal))
    noisy_signal = clean_signal + noise_level * noise
    print(f"✅ Noise added at level: {noise_level}")
    return noisy_signal


def save_audio(signal, sr, filepath):
    """Save a numpy array as a .wav file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sf.write(filepath, signal, sr)
    print(f"✅ Saved: {filepath}")


def preprocess(input_path, noisy_output_path, noise_level=0.3):
    """Full preprocessing pipeline"""
    print("\n--- Starting Preprocessing ---")
    
    # Step 1: Load clean audio
    clean, sr = load_audio(input_path)
    
    # Step 2: Add noise
    noisy = add_noise(clean, noise_level)
    
    # Step 3: Save noisy version
    save_audio(noisy, sr, noisy_output_path)
    
    print("\n--- Preprocessing Complete ---")
    return clean, noisy, sr


# Run directly to test
if __name__ == "__main__":
    preprocess(
        input_path="data/raw/clean_speech.wav",
        noisy_output_path="data/noisy/noisy_speech.wav",
        noise_level=0.3
    )