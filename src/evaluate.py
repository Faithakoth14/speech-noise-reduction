# src/evaluate.py

import librosa
import numpy as np
from pystoi import stoi


def calculate_snr(clean, processed):
    """Calculate Signal to Noise Ratio in dB"""
    min_len = min(len(clean), len(processed))
    clean = clean[:min_len]
    processed = processed[:min_len]
    noise = clean - processed
    snr = 10 * np.log10(np.var(clean) / (np.var(noise) + 1e-10))
    return round(snr, 2)


def calculate_stoi(clean, processed, sr=16000):
    """Calculate STOI score (0 to 1, higher = more intelligible)"""
    min_len = min(len(clean), len(processed))
    clean = clean[:min_len]
    processed = processed[:min_len]
    try:
        score = stoi(clean, processed, sr, extended=False)
        return round(score, 3)
    except Exception as e:
        return f"Error: {e}"


def evaluate_all(clean_path, noisy_path, spectral_path, wiener_path, sr=16000):
    """Run full evaluation on all methods"""

    print("\n========================================")
    print("     SPEECH NOISE REDUCTION EVALUATION")
    print("========================================\n")

    # Load all files
    clean,    _ = librosa.load(clean_path,    sr=sr)
    noisy,    _ = librosa.load(noisy_path,    sr=sr)
    spectral, _ = librosa.load(spectral_path, sr=sr)
    wiener,   _ = librosa.load(wiener_path,   sr=sr)

    # Calculate metrics
    results = {
        'Noisy Speech':         {'snr': calculate_snr(clean, noisy),
                                  'stoi': calculate_stoi(clean, noisy, sr)},
        'Spectral Subtraction': {'snr': calculate_snr(clean, spectral),
                                  'stoi': calculate_stoi(clean, spectral, sr)},
        'Wiener Filter':        {'snr': calculate_snr(clean, wiener),
                                  'stoi': calculate_stoi(clean, wiener, sr)},
    }

    # Print results table
    print(f"{'Method':<25} {'SNR (dB)':<15} {'STOI':<10}")
    print("-" * 50)
    for method, scores in results.items():
        print(f"{method:<25} {str(scores['snr']):<15} {str(scores['stoi']):<10}")

    print("\n--- Metric Guide ---")
    print("SNR  : Higher is better (dB). Positive = noise reduced.")
    print("STOI : Range 0 to 1. Higher = more intelligible speech.")

    return results


if __name__ == "__main__":
    evaluate_all(
        clean_path    = "data/raw/clean_speech.wav",
        noisy_path    = "data/noisy/noisy_speech.wav",
        spectral_path = "data/output/spectral_cleaned.wav",
        wiener_path   = "data/output/wiener_cleaned.wav"
    )