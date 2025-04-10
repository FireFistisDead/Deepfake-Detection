# 🔍 Fine-Tuning AASIST for Audio Deepfake Detection

This repository demonstrates the fine-tuning of the **AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Network)** model for detecting audio deepfakes. The model is trained on a subset of the ASVspoof2019 Logical Access (LA) dataset and showcases how transfer learning can be effectively used for synthetic speech detection with limited data.

---

## 🧠 Model Overview

AASIST is a cutting-edge architecture that operates directly on raw audio waveforms. It combines:
- Spectral and temporal convolutional encoding,
- Graph attention layers (GAT) to model time-frequency relationships,
- Learnable master nodes and pooling strategies.

Unlike traditional models that rely on pre-processed audio features (like MFCCs or spectrograms), AASIST captures more holistic and nuanced patterns, making it ideal for detecting synthetic audio.

---
