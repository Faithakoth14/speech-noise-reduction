# gui.py — Modern Speech Noise Reduction App (Before/After Playback)

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import shutil
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from scipy.signal import butter, filtfilt


# ── Colour tokens ──────────────────────────────────────────────
BG         = "#0F0F13"
SURFACE    = "#1A1A24"
SURFACE2   = "#22222F"
BORDER     = "#2E2E40"
ACCENT     = "#7C6EF5"
ACCENT_DIM = "#3D3580"
SUCCESS    = "#3DD68C"
DANGER     = "#F56E6E"
WARN       = "#F5A623"
TEXT_PRI   = "#F0EFF8"
TEXT_SEC   = "#8B8A9E"
TEXT_MUT   = "#55546A"
WHITE      = "#FFFFFF"


# ── Animated pulse bar ─────────────────────────────────────────
class PulseBar(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, height=3, highlightthickness=0, **kw)
        self._pos   = 0.0
        self._dir   = 1
        self._after = None

    def start(self):
        self._pos = 0.0
        self._tick()

    def stop(self):
        if self._after:
            self.after_cancel(self._after)
            self._after = None
        self.delete("all")

    def _tick(self):
        w = self.winfo_width() or 460
        span = int(w * 0.35)
        x0 = int(self._pos * (w - span))
        self.delete("all")
        self.create_rectangle(x0, 0, x0 + span, 3, fill=ACCENT, outline="")
        self._pos += 0.018 * self._dir
        if self._pos >= 1.0:
            self._pos, self._dir = 1.0, -1
        elif self._pos <= 0.0:
            self._pos, self._dir = 0.0, 1
        self._after = self.after(22, self._tick)


# ── SNR helper ─────────────────────────────────────────────────
def calc_snr(original, processed):
    n    = min(len(original), len(processed))
    diff = original[:n] - processed[:n]
    return round(10 * np.log10(np.var(original[:n]) / (np.var(diff) + 1e-10)), 1)


# ── Bandpass filter ────────────────────────────────────────────
def bandpass(y, sr, lo=80, hi=7000):
    nyq = sr / 2
    b, a = butter(4, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, y).astype(np.float32)


# ── App ────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DeNoise — Speech Clarity Tool")
        self.geometry("520x740")
        self.resizable(False, False)
        self.configure(bg=BG)

        self.input_path  = None
        self.output_path = None
        self.y           = None
        self.sr          = None
        self.reduced     = None
        self._playing    = None  # track which is playing: 'before' or 'after'

        try:
            import pygame
            pygame.mixer.init()
            self._pygame = pygame
        except Exception:
            self._pygame = None

        self._build()

    # ── Build UI ──────────────────────────────────────────────
    def _build(self):

        # ── HEADER ──────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(20, 0))
        tk.Label(hdr, text="DeNoise",
                 font=("Helvetica", 22, "bold"),
                 bg=BG, fg=TEXT_PRI).pack(side="left")
        tk.Label(hdr, text="Speech Clarity Tool",
                 font=("Helvetica", 10),
                 bg=BG, fg=TEXT_MUT).pack(side="left", padx=(10, 0), pady=(6, 0))

        self._divider(pady=(12, 0))

        # ── INPUT FILE ───────────────────────────────────────
        file_card = self._card()
        file_row  = tk.Frame(file_card, bg=SURFACE)
        file_row.pack(fill="x", padx=14, pady=10)

        self._file_icon = tk.Label(file_row, text="⬆",
                                   font=("Helvetica", 16),
                                   bg=SURFACE, fg=ACCENT)
        self._file_icon.pack(side="left", padx=(0, 10))

        info = tk.Frame(file_row, bg=SURFACE)
        info.pack(side="left", fill="x", expand=True)

        self._file_name = tk.Label(info, text="No file selected",
                                   font=("Helvetica", 10, "bold"),
                                   bg=SURFACE, fg=TEXT_PRI, anchor="w")
        self._file_name.pack(anchor="w")
        self._file_meta = tk.Label(info, text="Accepts  .wav  .flac  .mp3",
                                   font=("Helvetica", 8),
                                   bg=SURFACE, fg=TEXT_MUT, anchor="w")
        self._file_meta.pack(anchor="w")

        tk.Button(file_row, text="Browse", command=self._browse,
                  bg=ACCENT, fg=WHITE,
                  font=("Helvetica", 9, "bold"),
                  relief="flat", bd=0,
                  activebackground=ACCENT_DIM,
                  activeforeground=WHITE,
                  padx=12, pady=5, cursor="hand2").pack(side="right")

        self._divider()

        # ── ALGORITHM ────────────────────────────────────────
        algo_card = self._card()
        algo_row  = tk.Frame(algo_card, bg=SURFACE)
        algo_row.pack(fill="x", padx=14, pady=10)

        badge = tk.Frame(algo_row, bg=ACCENT_DIM)
        badge.pack(side="left", padx=(0, 10))
        tk.Label(badge, text="3x",
                 font=("Helvetica", 10, "bold"),
                 bg=ACCENT_DIM, fg=ACCENT,
                 padx=8, pady=6).pack()

        algo_info = tk.Frame(algo_row, bg=SURFACE)
        algo_info.pack(side="left")
        tk.Label(algo_info, text="Multi-Pass Spectral Subtraction",
                 font=("Helvetica", 10, "bold"),
                 bg=SURFACE, fg=TEXT_PRI).pack(anchor="w")
        tk.Label(algo_info, text="3 aggressive passes — maximum noise removal",
                 font=("Helvetica", 8),
                 bg=SURFACE, fg=TEXT_MUT).pack(anchor="w")

        pills = tk.Frame(algo_card, bg=SURFACE)
        pills.pack(anchor="w", padx=14, pady=(0, 10))
        for label in ["Pass 1: Non-stationary", "Pass 2: Stationary", "Pass 3: Residual"]:
            p = tk.Frame(pills, bg=SURFACE2,
                         highlightbackground=BORDER,
                         highlightthickness=1)
            p.pack(side="left", padx=(0, 5))
            tk.Label(p, text=label, font=("Helvetica", 7),
                     bg=SURFACE2, fg=TEXT_SEC,
                     padx=6, pady=3).pack()

        self._divider()

        # ── RUN BUTTON ───────────────────────────────────────
        self._run_btn = tk.Button(
            self, text="⚡  Apply Noise Reduction",
            command=self._process,
            bg=ACCENT, fg=WHITE,
            font=("Helvetica", 11, "bold"),
            relief="flat", bd=0,
            activebackground=ACCENT_DIM,
            activeforeground=WHITE,
            pady=12, cursor="hand2"
        )
        self._run_btn.pack(fill="x", padx=24)

        self._pulse = PulseBar(self, bg=BG)
        self._pulse.pack(fill="x", padx=24, pady=(6, 0))

        self._status = tk.Label(self,
                                text="Upload a file to get started.",
                                font=("Helvetica", 9),
                                bg=BG, fg=TEXT_MUT)
        self._status.pack(anchor="w", padx=24, pady=(4, 0))

        self._divider()

        # ── BEFORE / AFTER PLAYBACK ───────────────────────────
        ba_label = tk.Label(self, text="BEFORE / AFTER",
                            font=("Helvetica", 8, "bold"),
                            bg=BG, fg=TEXT_MUT)
        ba_label.pack(anchor="w", padx=24, pady=(0, 8))

        ba_card = self._card()
        ba_inner = tk.Frame(ba_card, bg=SURFACE)
        ba_inner.pack(fill="x", padx=14, pady=12)

        # Before block
        before_block = tk.Frame(ba_inner, bg=SURFACE2,
                                highlightbackground=BORDER,
                                highlightthickness=1)
        before_block.pack(side="left", fill="both", expand=True,
                          padx=(0, 6))

        tk.Label(before_block, text="BEFORE",
                 font=("Helvetica", 8, "bold"),
                 bg=SURFACE2, fg=WARN).pack(pady=(8, 2))
        tk.Label(before_block, text="Original noisy audio",
                 font=("Helvetica", 7),
                 bg=SURFACE2, fg=TEXT_MUT).pack()

        self._before_btn = tk.Button(
            before_block, text="▶  Play",
            command=self._play_before,
            bg=WARN, fg="#1e1e2e",
            font=("Helvetica", 9, "bold"),
            relief="flat", bd=0,
            activebackground="#c4861c",
            activeforeground=WHITE,
            padx=10, pady=6, cursor="hand2"
        )
        self._before_btn.pack(pady=(6, 8), padx=10, fill="x")

        # After block
        after_block = tk.Frame(ba_inner, bg=SURFACE2,
                               highlightbackground=BORDER,
                               highlightthickness=1)
        after_block.pack(side="left", fill="both", expand=True,
                         padx=(6, 0))

        tk.Label(after_block, text="AFTER",
                 font=("Helvetica", 8, "bold"),
                 bg=SURFACE2, fg=SUCCESS).pack(pady=(8, 2))
        tk.Label(after_block, text="Cleaned speech audio",
                 font=("Helvetica", 7),
                 bg=SURFACE2, fg=TEXT_MUT).pack()

        self._after_btn = tk.Button(
            after_block, text="▶  Play",
            command=self._play_after,
            bg=SUCCESS, fg="#1e1e2e",
            font=("Helvetica", 9, "bold"),
            relief="flat", bd=0,
            activebackground="#2aad72",
            activeforeground=WHITE,
            padx=10, pady=6, cursor="hand2"
        )
        self._after_btn.pack(pady=(6, 8), padx=10, fill="x")

        # Stop button
        stop_btn = tk.Button(
            ba_card, text="⏹  Stop Playback",
            command=self._stop,
            bg=SURFACE, fg=TEXT_SEC,
            font=("Helvetica", 8),
            relief="flat", bd=0,
            activebackground=SURFACE2,
            activeforeground=TEXT_PRI,
            pady=4, cursor="hand2"
        )
        stop_btn.pack(pady=(0, 8))

        self._divider()

        # ── RESULTS ──────────────────────────────────────────
        res_label = tk.Label(self, text="RESULTS",
                             font=("Helvetica", 8, "bold"),
                             bg=BG, fg=TEXT_MUT)
        res_label.pack(anchor="w", padx=24, pady=(0, 8))

        res_card  = self._card()
        res_inner = tk.Frame(res_card, bg=SURFACE)
        res_inner.pack(fill="x", padx=14, pady=12)

        # SNR row
        snr_row = tk.Frame(res_inner, bg=SURFACE)
        snr_row.pack(fill="x", pady=(0, 6))

        b = tk.Frame(snr_row, bg=SURFACE)
        b.pack(side="left", padx=(0, 16))
        self._snr_before_val = tk.Label(b, text="—",
                                        font=("Helvetica", 18, "bold"),
                                        bg=SURFACE, fg=WARN)
        self._snr_before_val.pack(anchor="w")
        tk.Label(b, text="SNR before",
                 font=("Helvetica", 8),
                 bg=SURFACE, fg=TEXT_MUT).pack(anchor="w")

        tk.Label(snr_row, text="→",
                 font=("Helvetica", 14),
                 bg=SURFACE, fg=TEXT_MUT).pack(side="left", padx=(0, 16))

        a = tk.Frame(snr_row, bg=SURFACE)
        a.pack(side="left")
        self._snr_after_val = tk.Label(a, text="—",
                                       font=("Helvetica", 18, "bold"),
                                       bg=SURFACE, fg=SUCCESS)
        self._snr_after_val.pack(anchor="w")
        tk.Label(a, text="SNR after",
                 font=("Helvetica", 8),
                 bg=SURFACE, fg=TEXT_MUT).pack(anchor="w")

        self._improve_tag = tk.Label(res_inner, text="",
                                     font=("Helvetica", 9, "bold"),
                                     bg=SURFACE, fg=SUCCESS)
        self._improve_tag.pack(anchor="w", pady=(2, 6))

        self._out_label = tk.Label(res_inner, text="Output: —",
                                   font=("Helvetica", 8),
                                   bg=SURFACE, fg=TEXT_MUT, anchor="w")
        self._out_label.pack(anchor="w", pady=(0, 10))

        # Save button
        self._mk_ghost(res_inner, "↓  Save cleaned file", self._download).pack(anchor="w")

    # ── Helpers ───────────────────────────────────────────────
    def _divider(self, pady=16):
        if isinstance(pady, int):
            pady = (pady, pady)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24, pady=pady)

    def _card(self):
        f = tk.Frame(self, bg=SURFACE,
                     highlightbackground=BORDER,
                     highlightthickness=1)
        f.pack(fill="x", padx=24, pady=(0, 2))
        return f

    def _mk_ghost(self, parent, text, cmd):
        f = tk.Frame(parent, bg=SURFACE2,
                     highlightbackground=BORDER,
                     highlightthickness=1)
        tk.Button(f, text=text, command=cmd,
                  bg=SURFACE2, fg=TEXT_SEC,
                  font=("Helvetica", 9),
                  relief="flat", bd=0,
                  activebackground=SURFACE,
                  activeforeground=TEXT_PRI,
                  padx=12, pady=6,
                  cursor="hand2").pack()
        return f

    def _set_status(self, msg, color=TEXT_MUT):
        self._status.config(text=msg, fg=color)

    # ── Actions ───────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio", "*.wav *.flac *.mp3"), ("All", "*.*")]
        )
        if not path:
            return
        self.input_path = path
        self._set_status("Loading file...", TEXT_SEC)
        self.y, self.sr = librosa.load(path, sr=16000)
        dur = len(self.y) / self.sr
        self._file_name.config(text=os.path.basename(path))
        self._file_meta.config(
            text="{:.1f}s  ·  {}kHz  ·  mono".format(dur, self.sr // 1000))
        self._file_icon.config(text="✔", fg=SUCCESS)
        self._set_status("File loaded — click Apply Noise Reduction.", SUCCESS)

    def _process(self):
        if not self.input_path:
            messagebox.showwarning("No file", "Please select an audio file first.")
            return
        threading.Thread(target=self._run_processing, daemon=True).start()

    def _run_processing(self):
        self._pulse.start()
        self._run_btn.config(state="disabled", text="Processing...")

        try:
            # Bandpass
            def bp(y):
                nyq = self.sr / 2
                b, a = butter(4, [80/nyq, 7000/nyq], btype='band')
                return filtfilt(b, a, y).astype(np.float32)

            # Pass 1
            self._set_status("Pass 1 — non-stationary sweep...", TEXT_SEC)
            y_bp = bp(self.y)
            noise_sample = y_bp[:self.sr // 2]
            pass1 = nr.reduce_noise(
                y=y_bp, sr=self.sr,
                y_noise=noise_sample,
                prop_decrease=0.75,
                stationary=False,
                n_fft=1024,
                n_std_thresh_stationary=1.8
            )

            # Pass 2
            self._set_status("Pass 2 — stationary refinement...", TEXT_SEC)
            pass2 = nr.reduce_noise(
                y=pass1, sr=self.sr,
                stationary=True,
                prop_decrease=0.65,
                n_std_thresh_stationary=1.5
            )

            # Final bandpass
            self._set_status("Final polish...", TEXT_SEC)
            self.reduced = bp(pass2)

            os.makedirs("data/output", exist_ok=True)
            self.output_path = "data/output/gui_cleaned.wav"
            sf.write(self.output_path, self.reduced, self.sr)

            snr_before  = calc_snr(self.y, self.y)
            snr_after   = calc_snr(self.y, self.reduced)
            improvement = round(snr_after - snr_before, 1)

            self._snr_before_val.config(text="{:+.1f} dB".format(snr_before))
            self._snr_after_val.config(text="{:+.1f} dB".format(snr_after))
            self._improve_tag.config(
                text="▲ {:+.1f} dB improvement".format(improvement))
            self._out_label.config(text="Output: {}".format(self.output_path))
            self._set_status("✔  Done! Use Before/After to compare.", SUCCESS)

        except Exception as e:
            self._set_status("Error: {}".format(e), DANGER)
        finally:
            self._pulse.stop()
            self._run_btn.config(state="normal", text="⚡  Apply Noise Reduction")

    def _play_audio(self, path, label):
        if not self._pygame:
            os.startfile(path)
            return
        self._pygame.mixer.music.load(path)
        self._pygame.mixer.music.play()
        self._set_status("▶  Playing {} audio...".format(label), ACCENT)
        self._playing = label

    def _play_before(self):
        if not self.input_path:
            messagebox.showwarning("No file", "Please upload a file first.")
            return
        self._play_audio(self.input_path, "original (noisy)")
        self._before_btn.config(text="▶  Playing...")
        self._after_btn.config(text="▶  Play")
        self.after(2000, lambda: self._before_btn.config(text="▶  Play"))

    def _play_after(self):
        if not self.output_path:
            messagebox.showwarning("Not ready", "Please process the audio first.")
            return
        self._play_audio(self.output_path, "cleaned")
        self._after_btn.config(text="▶  Playing...")
        self._before_btn.config(text="▶  Play")
        self.after(2000, lambda: self._after_btn.config(text="▶  Play"))

    def _stop(self):
        if self._pygame:
            self._pygame.mixer.music.stop()
            self._set_status("⏹  Playback stopped.", TEXT_MUT)
            self._before_btn.config(text="▶  Play")
            self._after_btn.config(text="▶  Play")

    def _download(self):
        if not self.output_path:
            messagebox.showwarning("Not ready", "Process a file first.")
            return
        save = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
            initialfile="cleaned_speech.wav",
            title="Save cleaned audio"
        )
        if save:
            shutil.copy(self.output_path, save)
            self._set_status("✔  Saved — {}".format(os.path.basename(save)), SUCCESS)


if __name__ == "__main__":
    app = App()
    app.mainloop()
