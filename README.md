# Audio Denoising with CBAM-Enhanced UNet and Adaptive Hybrid Loss

A deep learning framework for audio denoising that combines a CBAM (Convolutional Block Attention Module) enhanced UNet architecture with an adaptive MSE+SSIM hybrid loss function. This project provides comprehensive comparisons against both traditional and state-of-the-art deep learning baselines.

## Features

- **CBAM-Enhanced UNet**: UNet architecture augmented with Channel Attention and Spatial Attention modules for improved feature selection
- **Adaptive Hybrid Loss**: Cosine-annealing scheduled combination of MSE and SSIM losses (α decays from 0.15 → 0.05)
- **Multiple Noise Types**: White, Pink, Brown, and Band-limited noise generation for training augmentation
- **Comprehensive Baselines**: Spectral Subtraction and multiple UNet ablation variants (with/without CBAM, different loss functions)
- **Full Evaluation Suite**: SNR, SI-SDR, PESQ, STOI metrics with statistical analysis
- **Visualization**: Loss curves, spectrogram comparison, PSD analysis, scatter plots

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-username/CBAM-ASAL.git
cd CBAM-ASAL
pip install -r requirements.txt

# 2. Prepare data (see Data Preparation below)

# 3. Train model
python train.py

# 4. Evaluate and generate results
python evaluate.py

# 5. Demo: Denoise your own audio
python demo_inference.py --input your_audio.wav --output denoised_audio.wav
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU (recommended) or AMD/Intel GPU or CPU

### Step-by-Step Installation

```bash
# Clone the repository
git clone https://github.com/your-username/CBAM-ASAL.git
cd CBAM-ASAL

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Hardware Support

The framework automatically detects and uses the best available compute device:

| Priority | Device | Detection |
|----------|--------|-----------|
| 1 | DirectML (AMD/Intel GPUs) | `torch-directml` |
| 2 | CUDA (NVIDIA GPUs) | `torch.cuda` |
| 3 | CPU | Fallback |

---

## Data Preparation

### Download Datasets

This project uses three audio domains. Download and place them as follows:

```
data/
├── music_clean/
│   └── archive/Data/
│       └── genres_original/    # GTZAN Dataset
│           ├── blues/
│           ├── classical/
│           ├── country/
│           └── ...             # 10 genres
├── ambient_clean/
│   └── ESC-50-master/
│       └── ESC-50-master/
│           └── audio/          # ESC-50 Dataset
│               ├── 1-137-A-0.wav
│               └── ...
└── speech_clean/
    └── LibriSpeech/            # LibriSpeech Dataset
        └── train-clean-100/
            ├── 19/
            ├── 84/
            └── ...
```

**Dataset Download Links:**
- **GTZAN Music**: http://marsyas.info/downloads/datasets.html
- **ESC-50**: https://github.com/karoldvl/ESC-50
- **LibriSpeech**: https://www.openslr.org/12/

### Supported Formats

`.wav`, `.flac` (automatically resampled to 16 kHz)

### Data Augmentation

During training, synthetic noise is added at random SNR levels (-5 dB to 20 dB):
- White noise
- Pink noise
- Brown noise
- Band-limited noise

---

## Usage

### Training

Train all ablation model variants:

```bash
python train.py
```

**Models trained:**
| Model | Description |
|-------|-------------|
| UNet (No CBAM) | Baseline UNet without attention |
| UNet (CBAM+MSE) | UNet with CBAM, MSE loss |
| UNet (CBAM+SSIM) | UNet with CBAM, SSIM loss |
| UNet (CBAM+Mix) | UNet with CBAM, fixed hybrid loss |
| UNet (NoCBAM+Mix) | UNet without CBAM, adaptive hybrid loss |
| **Ours** | UNet with CBAM + Adaptive Hybrid Loss |

Checkpoints saved to `checkpoints/`.

### Evaluation

Evaluate all models and generate comparison figures:

```bash
python evaluate.py
```

**Output in `results/`:**
- `full_metrics_results.txt` — All metrics comparison
- `statistical_significance_tests.txt` — p-value analysis
- `metrics_bar_chart_*.svg` — Bar charts by domain
- `spec_comparison_improved.svg` — Spectrogram comparison
- `rtf_measurements.txt` — Real-time factor & latency

### Demo: Denoise Your Own Audio

For reviewers to test the model on their own audio:

```bash
# Basic usage
python demo_inference.py --input noisy_audio.wav --output clean_audio.wav

# Use a specific checkpoint
python demo_inference.py -i input.wav -o output.wav -m checkpoints/best_model_cbam_ssim.pth
```

**Options:**
- `--input, -i`: Path to input audio file (supports .wav, .mp3, .flac, .ogg)
- `--output, -o`: Path to save denoised audio
- `--model, -m`: Path to model checkpoint (default: `checkpoints/best_model_cbam_ssim.pth`)

---

## Project Structure

```
CBAM-ASAL/
├── config.py                  # Global configuration
├── train.py                   # Training entry point
├── evaluate.py                # Evaluation entry point
├── demo_inference.py          # Demo inference for reviewers
├── requirements.txt           # Python dependencies
│
├── models/                    # Neural network architectures
│   ├── cbam.py               # CBAM attention mechanism
│   ├── unet.py                # UNet variants
│   └── losses.py              # Loss functions
│
├── data/                      # Data loading & preprocessing
│   ├── dataset.py             # AudioDataset class
│   └── noise.py               # Noise generation
│
├── baselines/                 # Baseline models
│   └── spectral_subtraction.py
│
├── utils/                     # Utilities
│   ├── metrics.py            # Evaluation metrics
│   ├── inference.py           # Inference functions
│   └── misc.py                # Misc utilities
│
├── visualization/             # Plotting
│   └── plots.py
│
├── checkpoints/               # Model weights (auto-created)
└── results/                   # Output results (auto-created)
```

---

## Model Architecture

### CBAM Attention Block

Sequential channel and spatial attention:

```
Input → Channel Attention → Spatial Attention → Output
           ↓                      ↓
      AvgPool+MaxPool        Conv7x7+Sigmoid
      Shared MLP
```

### UNet with CBAM

```
Encoder:  1→32→64→128→256→512 (MaxPool)
Decoder:  512→256→128→64→32→1 (TransposeConv + CBAM + skip)
```

### Adaptive Hybrid Loss

```
L = α(t)·L_MSE + (1-α(t))·(1-L_SSIM)

α(t) = 0.05 + (0.15-0.05)·½(1+cos(πt/T))
       ↑ final   ↑ initial    cosine annealing
```

---

## Evaluation Metrics

| Metric | Full Name | Range | Direction |
|--------|-----------|-------|-----------|
| SNR | Signal-to-Noise Ratio | -∞ to +∞ | Higher is better |
| SI-SDR | Scale-Invariant SDR | -∞ to +∞ | Higher is better |
| PESQ | Perceptual Evaluation of Speech Quality | -0.5 to 4.5 | Higher is better |
| STOI | Short-Time Objective Intelligibility | 0 to 1 | Higher is better |

---

## For Reviewers

If you received this code for review:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download pretrained weights:**
   The pretrained model `best_model_cbam_ssim.pth` should be in the `checkpoints/` directory.

3. **Test on your own audio:**
   ```bash
   python demo_inference.py --input your_audio.wav --output result.wav
   ```

4. **View quantitative results:**
   See `results/full_metrics_results.txt` for detailed metrics comparison.

---

## Citation

```bibtex
@article{cbam_asal_audio_denoising,
  title={Audio Denoising with CBAM-Enhanced UNet and Adaptive Hybrid Loss},
  author={Your Name},
  journal={Your Journal/Conference},
  year={2026}
}
```

---

## License

MIT License

## Acknowledgments

- [PyTorch](https://pytorch.org/) — Deep learning framework
- [GTZAN](https://marsyas.info/downloads/datasets.html), [ESC-50](https://github.com/karolpiczak/ESC-50), [LibriSpeech](https://www.openslr.org/12) — Audio datasets
