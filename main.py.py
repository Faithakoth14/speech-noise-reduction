# main.py

import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import noisereduce as nr
from scipy.signal import wiener
from pystoi import stoi


# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
CLEAN_PATH    = "data/raw/clean_speech.wav"
NOISY_PATH    = "data/noisy/noisy_speech.wav"
SPECTRAL_PATH = "data/output/spectral_cleaned.wav"
WIENER_PATH   = "data/output/wiener_cleaned.wav"
OUTPUT_DIR    = "data/output"
SAMPLE_RATE   = 16000
NOISE_LEVEL   = 0.3


# ─────────────────────────────────────────
#  STEP 1: PREPROCESS
# ─────────────────────────────────────────
def preprocess():
    print("\n[1/4] Preprocessing...")
    os.makedirs("data/noisy", exist_ok=True)
    os.makedirs(OUTPUT_DIR,   exist_ok=True)

    clean, sr = librosa.load(CLEAN_PATH, sr=SAMPLE_RATE)
    noise     = np.random.normal(0, 1, len(clean))
    noisy     = clean + NOISE_LEVEL * noise

    sf.write(NOISY_PATH, noisy, sr)
    print(f"   ✅ Noisy file saved: {NOISY_PATH}")
    return clean, noisy, sr


# ─────────────────────────────────────────
#  STEP 2: NOISE REDUCTION
# ─────────────────────────────────────────
def reduce_noise(noisy, sr):
    print("\n[2/4] Applying Noise Reduction...")

    # Spectral Subtraction
    noise_sample = noisy[:sr // 2]
    spectral = nr.reduce_noise(y=noisy, sr=sr, y_noise=noise_sample, prop_decrease=1.0)
    sf.write(SPECTRAL_PATH, spectral, sr)
    print("   ✅ Spectral Subtraction complete")

    # Wiener Filter
    wiener_cleaned = wiener(noisy, mysize=29).astype(np.float32)
    sf.write(WIENER_PATH, wiener_cleaned, sr)
    print("   ✅ Wiener Filter complete")

    return spectral, wiener_cleaned


# ─────────────────────────────────────────
#  STEP 3: EVALUATE
# ─────────────────────────────────────────
def calculate_snr(clean, processed):
    min_len = min(len(clean), len(processed))
    clean, processed = clean[:min_len], processed[:min_len]
    noise = clean - processed
    return round(10 * np.log10(np.var(clean) / (np.var(noise) + 1e-10)), 2)


def calculate_stoi(clean, processed, sr):
    min_len = min(len(clean), len(processed))
    return round(stoi(clean[:min_len], processed[:min_len], sr, extended=False), 3)


def evaluate(clean, noisy, spectral, wiener_cleaned, sr):
    print("\n[3/4] Evaluating...")
    print(f"\n{'Method':<25} {'SNR (dB)':<15} {'STOI':<10}")
    print("-" * 50)

    results = {
        "Noisy Speech":         (noisy,          ),
        "Spectral Subtraction": (spectral,        ),
        "Wiener Filter":        (wiener_cleaned,  ),
    }

    for method, (signal,) in results.items():
        snr  = calculate_snr(clean, signal)
        stoi_score = calculate_stoi(clean, signal, sr)
        print(f"   {method:<22} {str(snr):<15} {str(stoi_score):<10}")


# ─────────────────────────────────────────
#  STEP 4: VISUALIZE
# ─────────────────────────────────────────
def visualize(clean, noisy, spectral, wiener_cleaned, sr):
    print("\n[4/4] Generating Visualizations...")

    signals = [clean, noisy, spectral, wiener_cleaned]
    titles  = [
        "1. Clean Speech",
        "2. Noisy Speech",
        "3. Spectral Subtraction",
        "4. Wiener Filter"
    ]
    colors = ["blue", "red", "green", "purple"]

    # Waveforms
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    for i, (sig, title, color) in enumerate(zip(signals, titles, colors)):
        librosa.display.waveshow(sig, sr=sr, ax=axes[i], color=color)
        axes[i].set_title(title)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/final_waveforms.png")
    plt.close()
    print(f"   ✅ Waveforms saved")

    # Spectrograms
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    for i, (sig, title) in enumerate(zip(signals, titles)):
        D   = librosa.stft(sig)
        img = librosa.display.specshow(
            librosa.amplitude_to_db(abs(D), ref=np.max),
            sr=sr, x_axis='time', y_axis='hz', ax=axes[i]
        )
        axes[i].set_title(title)
        fig.colorbar(img, ax=axes[i], format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/final_spectrograms.png")
    plt.close()
    print(f"   ✅ Spectrograms saved")


# ─────────────────────────────────────────
#  RUN FULL PIPELINE
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   SPEECH NOISE REDUCTION PIPELINE")
    print("=" * 50)

    clean, noisy, sr               = preprocess()
    spectral, wiener_cleaned       = reduce_noise(noisy, sr)
    evaluate(clean, noisy, spectral, wiener_cleaned, sr)
    visualize(clean, noisy, spectral, wiener_cleaned, sr)

    print("\n" + "=" * 50)
    print("   ✅ PIPELINE COMPLETE!")
    print(f"   📁 All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 50)