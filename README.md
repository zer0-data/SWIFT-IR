# SWIFT-IR: Smart Weather Inference For Thermal Images

**SWIFT-IR** is a dynamic, weather-aware object detection pipeline designed for thermal (IR) imagery. It utilizes a 3-stage conditional routing mechanism to optimize performance in Clear, Foggy, and Rainy conditions, ensuring high accuracy with minimal latency.

## 🚀 Pipeline Overview

### Stage A: Classification & Routing
- **Input:** 14-Bit RAW Thermal Stream.
- **Classifier:** 3-Layer MLP using statistical features (Variance, Entropy, MinMax).
- **Action:** - *Clear:* Zero-latency pass-through.
  - *Fog:* Gamma Correction.
  - *Rain:* Lightweight Spatial Residual Block (LSRB) for streak removal.

### Stage B: Preprocessing
- **Fog:** Adaptive Gamma Correction based on scene mean intensity.
- **Rain:** Deep learning-based rain mask subtraction.

### Stage C: Detection (Student-Teacher Distillation)
- **Backbone:** Modified YOLOv8-Small with **ViT Neck** (Global Context).
- **Training:** Cross-Modal Distillation.
  - *Teacher:* YOLOv8-Large trained on **Pseudo-RGB** (Generated via CycleGAN).
  - *Student:* YOLOv8-Small trained on **Raw IR**.
  - *Goal:* Force the student to "hallucinate" RGB features from IR input.



## 📂 Datasets Used
1. **FLIR ADAS v2:** Teacher training backbone.
2. **M3FD:** Paired RGB/Thermal for calibration and overfitting prevention.
3. **C3I:** Weather classification training.

## 🛠️ Installation

```bash
git clone [https://github.com/zer0_data/SWIFT-IR.git](https://github.com/zer0_data/SWIFT-IR.git)
cd SWIFT-IR
pip install -r requirements.txt
