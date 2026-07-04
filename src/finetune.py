# src/finetune.py
import librosa
import numpy as np
import soundfile as sf
import os
import random
import noisereduce as nr
from scipy.signal import butter, filtfilt, wiener
from pystoi import stoi


def mix_at_snr(clean, noise, target_snr_db):
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)]
    clean_power = np.var(clean)
    noise_power = np.var(noise)
    scale = np.sqrt(clean_power / (noise_power * 10 ** (target_snr_db / 10)))
    return (clean + scale * noise).astype(np.float32), (scale * noise).astype(np.float32)


def bandpass(y, sr, lo=80, hi=7000):
    nyq = sr / 2
    b, a = butter(4, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, y).astype(np.float32)


def spectral_subtraction(y, sr):
    y = bandpass(y, sr)
    noise_sample = y[:sr // 2]
    pass1 = nr.reduce_noise(
        y=y, sr=sr, y_noise=noise_sample,
        prop_decrease=0.75, stationary=False,
        n_fft=1024, n_std_thresh_stationary=1.8
    )
    pass2 = nr.reduce_noise(
        y=pass1, sr=sr, stationary=True,
        prop_decrease=0.65, n_std_thresh_stationary=1.5
    )
    return bandpass(pass2, sr)


def wiener_filter(y, sr):
    y = bandpass(y, sr)
    cleaned = wiener(y, mysize=29).astype(np.float32)
    return bandpass(cleaned, sr)


def calc_snr(clean, noise):
    n = min(len(clean), len(noise))
    return round(10 * np.log10(np.var(clean[:n]) / (np.var(noise[:n]) + 1e-10)), 1)


def calc_stoi(clean, cleaned, sr):
    n = min(len(clean), len(cleaned))
    return round(stoi(clean[:n], cleaned[:n], sr, extended=False), 3)


def calc_snr_output(noisy, cleaned):
    n = min(len(noisy), len(cleaned))
    residual = noisy[:n] - cleaned[:n]
    return round(10 * np.log10(np.var(cleaned[:n]) / (np.var(residual) + 1e-10)), 1)


def run_algorithm(clean, selected, noise_dir, output_dir, snr_levels, sr, algo_fn, tag):
    results = []
    for noise_file in selected:
        noise_name = noise_file.split('_')[0][:18]
        noise, _ = librosa.load(os.path.join(noise_dir, noise_file), sr=sr)
        for snr_db in snr_levels:
            noisy, noise_sig = mix_at_snr(clean, noise, target_snr_db=snr_db)
            cleaned = algo_fn(noisy, sr)
            snr_in   = calc_snr(clean, noise_sig)
            snr_out  = calc_snr_output(noisy, cleaned)
            gain     = round(snr_out - snr_in, 1)
            stoi_val = calc_stoi(clean, cleaned, sr)
            if stoi_val >= 0.90:
                quality = "Excellent"
            elif stoi_val >= 0.75:
                quality = "Good"
            elif stoi_val >= 0.60:
                quality = "Fair"
            else:
                quality = "Poor"
            gain_str = "+{:.1f}".format(gain) if gain >= 0 else "{:.1f}".format(gain)
            print("{:<20} {:>7.1f} {:>9.1f} {:>8} {:>7.3f}   {}".format(
                noise_name, snr_in, snr_out, gain_str, stoi_val, quality))
            results.append(stoi_val)
            if snr_db == 10:
                sf.write("{}/{}_{}_noisy.wav".format(output_dir, noise_name, tag), noisy, sr)
                sf.write("{}/{}_{}_cleaned.wav".format(output_dir, noise_name, tag), cleaned, sr)
    return results


def finetune_evaluate(
    clean_path="data/raw/clean_speech.wav",
    noise_dir="MS-SNSD/noise_train",
    output_dir="data/output/finetune",
    snr_levels=[5, 10, 15],
    sr=16000
):
    os.makedirs(output_dir, exist_ok=True)
    clean, _ = librosa.load(clean_path, sr=sr)

    noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]
    categories  = list(set(f.split('_')[0] for f in noise_files))
    selected    = []
    for cat in random.sample(categories, min(5, len(categories))):
        cat_files = [f for f in noise_files if f.startswith(cat)]
        selected.append(random.choice(cat_files))

    header = "Noise Type           SNR In   SNR Out     Gain    STOI   Quality"
    line   = "-" * 72

    print()
    print("=" * 72)
    print("   ALGORITHM 1: SPECTRAL SUBTRACTION (3-Pass)")
    print("=" * 72)
    print(header)
    print(line)
    ss_results = run_algorithm(clean, selected, noise_dir, output_dir, snr_levels, sr, spectral_subtraction, "ss")
    print(line)
    print("Average STOI: {:.3f}".format(sum(ss_results) / len(ss_results)))

    print()
    print("=" * 72)
    print("   ALGORITHM 2: WIENER FILTER")
    print("=" * 72)
    print(header)
    print(line)
    wf_results = run_algorithm(clean, selected, noise_dir, output_dir, snr_levels, sr, wiener_filter, "wf")
    print(line)
    print("Average STOI: {:.3f}".format(sum(wf_results) / len(wf_results)))

    ss_avg = sum(ss_results) / len(ss_results)
    wf_avg = sum(wf_results) / len(wf_results)
    winner = "Spectral Subtraction" if ss_avg > wf_avg else "Wiener Filter"

    print()
    print("=" * 72)
    print("   COMPARISON SUMMARY")
    print("=" * 72)
    print("Algorithm                    Avg STOI    Verdict")
    print(line)
    print("{:<28} {:.3f}       {}".format("Spectral Subtraction (3-Pass)", ss_avg, "WINNER" if winner == "Spectral Subtraction" else ""))
    print("{:<28} {:.3f}       {}".format("Wiener Filter", wf_avg, "WINNER" if winner == "Wiener Filter" else ""))
    print(line)
    print("Best Algorithm: {}".format(winner))
    print()
    print("Note: Negative SNR gain is expected with classical methods due")
    print("to musical noise artifacts (Boll, 1979). STOI is the primary")
    print("metric for speech intelligibility assessment.")
    print()
    print("Sample files saved to: {}".format(output_dir))
    print("=" * 72)


if __name__ == "__main__":
    finetune_evaluate()
