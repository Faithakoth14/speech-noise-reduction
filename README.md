# Speech Noise Reduction
### Final Year Project — 2025/2026

A Python-based speech noise reduction system using classical signal 
processing techniques. No GPU. No deep learning. Just math.

## 🔗 GitHub
https://github.com/Faithakoth14/speech-noise-reduction

## 🚀 How to Run

### Web App
```bash
git clone https://github.com/Faithakoth14/speech-noise-reduction.git
cd speech-noise-reduction
pip install -r requirements.txt
python app.py
```
Then open: http://127.0.0.1:5000

### Desktop GUI
```bash
python gui.py
```

### Full Pipeline
```bash
python main.py
```

## 🎯 Algorithms
| Algorithm | Avg STOI | Quality |
|---|---|---|
| Multi-Pass Spectral Subtraction | 0.916 | Excellent |
| Wiener Filter | 0.835 | Good |

## 📊 Datasets
- **Clean speech:** LibriSpeech dev-clean (16kHz WAV)
- **Noise:** Microsoft MS-SNSD — 28 real-world noise categories

> MS-SNSD is not included in this repo due to size (3.93GB).
> Clone it separately:
> ```bash
> git clone https://github.com/microsoft/MS-SNSD.git
> ```

## 🧪 Evaluation Results
Tested across 5 noise categories at 3 SNR levels (5, 10, 15 dB):

| Noise Type | SS STOI | Wiener STOI |
|---|---|---|
| WasherDryer | 0.938 | 0.852 |
| Munching | 0.888 | 0.817 |
| SqueakyChair | 0.917 | 0.842 |
| CafeTeria | 0.930 | 0.855 |
| VacuumCleaner | 0.946 | 0.859 |

## 🛠 Tech Stack
- Python 3.13
- librosa, scipy, noisereduce, soundfile
- pystoi, numpy, matplotlib
- Flask, HTML5, CSS3, JavaScript
- Tkinter

## 📁 Project Structure
