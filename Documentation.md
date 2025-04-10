#  Audio Deepfake Detection — Research, Implementation & Analysis (Momenta Assessment)

##  Dataset Selection: ASVspoof2019 LA

### Why this dataset?
- **Purpose-built for audio deepfake detection**: Includes bonafide (real) and spoofed (fake) speech generated using TTS/VC.
- **Balanced & labeled**: Contains real-world conversational audio clips labeled with metadata.
- **Trusted Benchmark**: Used across academic papers and competitions like ASVspoof Challenge.

---

##  Model Selection and Rationale

###  1. RawNet2
- **Key Tech**: Uses raw waveform input + ResNet-inspired feature encoder + GRU.
- **Why**: Minimal preprocessing, strong baseline with public weights.
- **Performance**: EER ~1.7% on ASVspoof2019 LA.
- **Limitation**: Computationally heavy.

###  2. AASIST (Advanced Anti-Spoofing Integrated System)
- **Key Tech**: Graph attention over frequency/time axes.
- **Why**: Recent SOTA, highly interpretable.
- **Performance**: EER ~1.5% on ASVspoof2021 LA.
- **Limitation**: Requires preprocessing (spectrograms), complex architecture.

###  3. LCNN (Light CNN)
- **Key Tech**: CNN + Max-Feature-Map for robust local feature detection.
- **Why**: Lightweight, fast to train, common baseline in spoof detection.
- **Performance**: EER ~3.3% (older but useful for comparison).
- **Limitation**: Lower accuracy on new generation attacks.

---

##  Why Other Models Were Excluded

- **X-vector + PLDA**: Strong for speaker ID, weak on deepfakes.
- **ResNet34/CRNN**: Require heavier tuning + augmentation.
- **TDNN**: Better suited to speaker ID than general spoof detection.

---

##  Implementation Overview

Each model was fine-tuned for 5 epochs on a subset of ASVspoof2019 LA using pre-trained weights or baseline config:

| Model    | Notebook Location |
|----------|------------------|
| RawNet2  | `finetune_model.py` |
| AASIST   | `yet-to-implement`  |
| LCNN     | `yet-to-implement`    |

Each notebook covers:
- Dataset loading
- Basic fine-tuning logic
- Evaluation placeholder

---

##  Observations & Strengths

| Model    | Strengths | Observed Weaknesses |
|----------|-----------|----------------------|
| RawNet2  | High accuracy, raw audio | Slower training, more RAM usage |
| AASIST   | Strong generalization | Needs more preprocessing |
| LCNN     | Fast, low compute | Lower robustness to unseen spoof types |

---

##  Reflections

### Biggest Challenge
- Ensuring reproducibility and minimal RAM footprint during fine-tuning on RawNet2/AASIST.

### Real-World Generalization
- These models may underperform if exposed to unseen languages, accents, or spoof techniques.
- Retraining/fine-tuning on new data is crucial.

### What More is Needed?
- Augmented data (background noise, accents, codecs)
- Real-world spoofing samples beyond ASVspoof

### Deployment Suggestions
- Use LCNN for edge (mobile) deployment.
- Use RawNet2/AASIST for server-side detection.
- Add audio pre-filtering pipeline (denoising, VAD).

