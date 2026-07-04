# src/noise_reduce.py

import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
import os

def reduce_noise_spectral(noisy_path, output_path, sr=16000):
    """Noise reduction using Spectral Subtraction - noisereduce library"""
    print("\n--- Spectral Subtraction ---")
    
    # Load noisy audio
    y, sr = librosa.load(noisy_path, sr=sr)
    
    # Use first 0.5 seconds as noise profile
    noise_sample = y[:sr // 2]
    
    # Apply noise reduction
    reduced = nr.reduce_noise(
        y=y,
        sr=sr,
        y_noise=noise_sample,
        prop_decrease=1.0
    )
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, reduced, sr)
    print(f"✅ Spectral subtraction complete!")
    print(f"✅ Saved to: {output_path}")
    return reduced, sr


def reduce_noise_wiener(noisy_path, output_path, sr=16000):
    """Noise reduction using Wiener Filter - scipy"""
    from scipy.signal import wiener
    print("\n--- Wiener Filter ---")
    
    # Load noisy audio
    y, sr = librosa.load(noisy_path, sr=sr)
    
    # Apply wiener filter
    reduced = wiener(y, mysize=29)
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, reduced.astype(np.float32), sr)
    print(f"✅ Wiener filter complete!")
    print(f"✅ Saved to: {output_path}")
    return reduced, sr


if __name__ == "__main__":
    # Run both methods
    reduce_noise_spectral(
        noisy_path="data/noisy/noisy_speech.wav",
        output_path="data/output/spectral_cleaned.wav"
    )
    
    reduce_noise_wiener(
        noisy_path="data/noisy/noisy_speech.wav",
        output_path="data/output/wiener_cleaned.wav"
    )