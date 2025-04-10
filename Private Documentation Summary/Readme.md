# Fine-Tuning AASIST, RawNet2, and LCNN for Audio Deepfake Detection

This repository documents the research and experimentation process of fine-tuning three advanced models—**AASIST**, **RawNet2**, and **LCNN**—on the ASVspoof2019 LA dataset for audio deepfake detection under limited-resource settings.

---

## 📌 Overview

With the rise of synthetic audio attacks, this work addresses the need for efficient, reliable, and lightweight deepfake detection systems. By fine-tuning three prominent models on a reduced dataset, the research demonstrates how well these models generalize in constrained environments.

---

## 🔍 Models Investigated

- **AASIST**  
  A graph attention-based model capturing both temporal and spectral cues from raw waveforms, fine-tuned on 1000 samples with cosine annealing and minimal epochs.

- **RawNet2**  
  An end-to-end deep neural network working directly on raw audio, employing residual CNNs, GRUs, and attention pooling for spoof detection.

- **LCNN**  
  A lightweight CNN that uses Max-Feature-Map activations to suppress redundant features and performs efficiently on log-mel spectrograms.

---

## 🧪 Experimental Setup

- Dataset: Subset of 1000 audio clips from **ASVspoof2019-LA**
- Optimizer: `Adam` for all models
- Loss: `CrossEntropyLoss`
- Schedulers:
  - AASIST & RawNet2: Cosine Annealing
  - LCNN: Step Decay
- Epochs:
  - AASIST: 5
  - RawNet2, LCNN: 10
- Batch Sizes:
  - AASIST & LCNN: 16
  - RawNet2: 8

---

## 📊 Results Summary

| Model     | Final Accuracy (%) | Final Loss (approx.) | Epochs |
|-----------|--------------------|-----------------------|--------|
| AASIST    | 45–60              | 5.75                  | 5      |
| RawNet2   | ~84.3              | 1.19                  | 10     |
| LCNN      | ~81.4              | 1.26                  | 10     |

---

## 💡 Novel Contributions

- Fine-tuning under **data-scarce conditions**
- **Pre-trained checkpoint utilization** for faster convergence
- **Simplified training pipelines** (no wrappers)
- **Raw waveform processing** (AASIST, RawNet2)
- **Efficient architecture design** for real-world deployment
- **Cross-model consistency** with common training configs

---

## 📚 References

- [AASIST: Graph Attention Network](https://arxiv.org/abs/2110.01200)
- [RawNet2: End-to-End Learning](https://arxiv.org/abs/2011.01108)
- [LCNN Anti-Spoofing](https://link.springer.com/article/10.1007/s42979-024-02774-9)
- [ASVspoof2019 Dataset](https://datashare.ed.ac.uk/handle/10283/3336)

> Additional surveys and references are listed in the full [Documentation PDF](./DEEPFAKE_Documentation.pdf)

---

## 🔭 Future Work

- Integration of **EER** and **min t-DCF** metrics
- Evaluation on **dev and eval splits**
- Visualization with **graph attention heatmaps**
- **Data augmentation** enhancements
- Cross-language and codec-based spoofing resistance

---

## 🗂️ License

This research is for academic and experimental purposes. For inquiries, please contact [Yash Khare](mailto:yashco.ltd@gmail.com).
