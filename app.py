# app.py
import os
import uuid
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, filtfilt
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

UPLOAD_FOLDER = "data/uploads"
OUTPUT_FOLDER = "data/output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def bandpass(y, sr, lo=80, hi=7000):
    nyq = sr / 2
    b, a = butter(4, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, y).astype(np.float32)


def normalize_audio(y, target_peak=0.9):
    """Normalize audio to a target peak level"""
    peak = np.max(np.abs(y))
    if peak > 0:
        return y * (target_peak / peak)
    return y


def denoise(y, sr):
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

    cleaned = bandpass(pass2, sr)
    cleaned = normalize_audio(cleaned, target_peak=0.9)
    return cleaned


def calc_snr(original, processed):
    n    = min(len(original), len(processed))
    diff = original[:n] - processed[:n]
    return float(round(10 * np.log10(np.var(original[:n]) / (np.var(diff) + 1e-10)), 1))


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file    = request.files["file"]
    file_id = str(uuid.uuid4())[:8]
    ext     = os.path.splitext(file.filename)[1]
    in_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_input{ext}")
    file.save(in_path)

    y, sr = librosa.load(in_path, sr=16000)
    dur   = round(len(y) / sr, 1)

    return jsonify({
        "file_id": file_id,
        "filename": file.filename,
        "duration": dur,
        "sample_rate": sr
    })


@app.route("/process", methods=["POST"])
def process():
    try:
        data    = request.get_json()
        file_id = data.get("file_id")

        # Find uploaded file
        uploads = os.listdir(UPLOAD_FOLDER)
        in_file = next((f for f in uploads if f.startswith(file_id)), None)

        if not in_file:
            return jsonify({"error": "File not found"}), 404

        in_path  = os.path.join(UPLOAD_FOLDER, in_file)
        out_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_cleaned.wav")

        print(f"Processing: {in_path}")
        print(f"Output to: {out_path}")

        y, sr = librosa.load(in_path, sr=16000)
        print(f"Loaded audio: {len(y)} samples at {sr}Hz")

        cleaned = denoise(y, sr)
        print("Denoising complete")

        sf.write(out_path, cleaned, sr)
        print(f"Saved to: {out_path}")

        snr_before  = calc_snr(y, y)
        snr_after   = calc_snr(y, cleaned)
        improvement = round(snr_after - snr_before, 1)

        return jsonify({
            "file_id":     file_id,
            "snr_before":  float(snr_before),
            "snr_after":   float(snr_after),
            "improvement": float(improvement)
        })

    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/audio/original/<file_id>")
def serve_original(file_id):
    uploads = os.listdir(UPLOAD_FOLDER)
    in_file = next((f for f in uploads if f.startswith(file_id)), None)
    if not in_file:
        return "Not found", 404
    return send_file(os.path.join(UPLOAD_FOLDER, in_file), mimetype="audio/wav")


@app.route("/audio/cleaned/<file_id>")
def serve_cleaned(file_id):
    for tier in ['free', 'pro']:
        out_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_{tier}_cleaned.wav")
        if os.path.exists(out_path):
            return send_file(out_path, mimetype="audio/wav")
    out_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_cleaned.wav")
    if os.path.exists(out_path):
        return send_file(out_path, mimetype="audio/wav")
    return "Not found", 404


@app.route("/download/<file_id>")
def download(file_id):
    out_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_cleaned.wav")
    if not os.path.exists(out_path):
        return "Not found", 404
    return send_file(out_path, as_attachment=True,
                     download_name="cleaned_speech.wav")


if __name__ == "__main__":
    app.run(debug=True)