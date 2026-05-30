# Radio Modulation Classification using FFT and Deep Learning

Classifies radio signal modulation types using FFT-extracted frequency features and a 1D convolutional neural network.

Built on the RadioML 2016.10a dataset.

---

## What it does

Each radio modulation scheme - BPSK, QPSK, QAM, FM and so on - leaves a distinct fingerprint in the frequency domain.

This project applies the Fourier transform to convert raw I/Q signals into magnitude spectrum, then trains a CNN to recognize those fingerprints automatically.

**Pipeline:** Raw I/Q signal -> FFT magnitude spectrum -> 1D CNN -> predicted modulation type

---

## Dataset

RadioML 2016.10a by DeepSig - 220,000 synthetic radio signal samples across 11 modulation types and 20 SNR levels (-20 dB to +18 dB).

The notebook downloads it automatically via kagglehub.

Manual download: https://www.kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom

---

## How to Run

**Google Colab (recommended - free GPU):**

1. Upload `radio_modulation_classifier.ipynb` to Colab
2. Set runtime to GPU: Runtime -> Change runtime type -> T4 GPU
3. Run all cells top to bottom

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook radio_modulation_classifier.ipynb
```

---

## Results

Overall test accuracy: **75%** on signals with SNR >= 0 dB

| Modulation | F1 Score | Notes |
|---|---|---|
| GFSK | 0.98 | Near-perfect |
| CPFSK | 0.98 | Near-perfect |
| AM-SSB | 0.97 | Near-perfect |
| PAM4 | 0.96 | Near-perfect |
| BPSK | 0.91 | Strong |
| AM-DSB | 0.77 | Good |
| QAM64 | 0.63 | Confused with QAM16 |
| WBFM | 0.59 | Confused with AM-DSB |
| 8PSK | 0.45 | Confused with QPSK |
| QAM16 | 0.40 | Confused with QAM64 |
| QPSK | 0.48 | Confused with 8PSK |

The model works well on modulations with distinct spectral shapes. The main failure is QPSK vs 8PSK and QAM16 vs QAM64 - these pairs have nearly identical FFT magnitude spectra and differ primarily in phase, which magnitude-only features cannot capture. This is a feature limitation, not a model problem.

---

## Load the Trained Model

```python
model = RadioClassifier()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()
```
