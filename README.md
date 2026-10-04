# DIF-SEG: Data-Driven Model for Medical Image Detection

## Overview
DIF-SEG is a deep learning-based Brain Tumor Detection and Segmentation system intended for academic and research purposes. It provides a robust, real-time end-to-end pipeline for detecting and dynamically outlining tumor boundaries within brain MRIs.

## Problem Statement
Accurate segmentation of brain tumors in MRI scans is critical for clinical analysis. Manual segmentation is highly time-consuming and subjective, necessitating a fast, automated, and mathematically stable AI approach for immediate region localization.

## Objectives
Provide an end-to-end pipeline for:
- Brain MRI Preprocessing & Quality Assessment
- High-Speed Tumor Detection
- Precise Pixel-Level Segmentation (Mask Generation)
- Bounding Box and Area Measurements
- Explainable AI Reporting

## Key Features
- **Real-time tumor prediction** with Streamlit
- **Bounding box extraction** around connected components
- **Tumor area percentage** calculations
- **Automated PDF Report** generation
- **Platform-independent** execution

## Unique Features
- **MRI Quality Gate**: Automatically rejects images with insufficient contrast or variance.
- **Segmentation Stability Test**: Mathematically compares boundary jitter by dynamically shifting thresholds to provide a Confidence/Stability Score.
- **Human-in-the-Loop Review**: Allows dynamic overriding and tuning of the AI's predicted boundaries via an active review slider without overwriting the core baseline.

## System Architecture
Upload MRI -> Quality Gate -> Preprocessing -> DIF-SEG Output -> Binary Mask Generation -> Connected Components Analysis -> Tumor Localization -> Human Review -> Export Report.

## Workflow
1. Start Application
2. Upload one MRI
3. Review Quality Gate
4. Run Inference
5. Review generated Boundary and Measurements
6. Approve or Correct Mask
7. Generate Final Report

## Technology Stack
- **Python 3.8+**
- **PyTorch** (Deep Learning)
- **Streamlit** (UI / Frontend)
- **OpenCV & Pillow** (Image Processing)
- **ReportLab** (PDF Generation)
- **Plotly** (Interactive MRI Viewer)

## Project Structure
```text
DIF-SEG-Brain-Tumor/
│
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── config.yaml
├── requirements.txt
├── README.md
├── PROJECT_ANALYSIS.md
├── PROJECT_CLEANUP.md
├── .gitignore
│
├── src/
├── scripts/
├── tests/
├── models/
├── dataset/
├── reports/
└── docs/
```

## Installation
Ensure Python 3.8+ is installed on your Windows machine.
```bash
git clone <repository_url>
cd DIF-SEG-Brain-Tumor
setup.bat
```

## Virtual Environment
The `setup.bat` file automatically generates a `venv` and installs dependencies.
To activate manually:
```bash
venv\Scripts\activate
```

## Dataset Setup
Download your desired dataset and extract it into the `dataset/` directory (e.g., `dataset/train`, `dataset/val`). The repository contains the structure but not the raw data itself due to medical privacy guidelines.

## Model Checkpoint Setup
The `models/hub_model.pth` file is bundled with the repository (31MB). No LFS or external downloading is required. Simply ensure it resides in the `models/` directory.

## Running the Application
```bash
python -m streamlit run app.py
```
Or simply use the included batch script:
```bash
run.bat
```

## Real-Time MRI Analysis
**MRI Upload**
↓
**MRI Quality Check**
↓
**DIF-SEG**
↓
**Tumor Segmentation**
↓
**Tumor Localization**
↓
**Tumor Geometry**
↓
**Stability Analysis**
↓
**Optional Human Review**
↓
**Report**

## Research Evaluation
Ground-truth masks are required **ONLY** for evaluation. The real-time MRI Analysis workflow requires no ground truth. The standalone Evaluation module uses uploaded Ground Truth arrays to compute clinical accuracy metrics.

## Evaluation Metrics
When Ground Truth is provided, the system calculates:
- **Dice Score**
- **IoU (Intersection over Union)**
- **Precision**
- **Recall**
- **F1 Score**
- **Pixel Accuracy**
- **Sensitivity**
- **Specificity**

## Screenshots
*(Add screenshots of the Streamlit Interface, Explainable AI tab, and PDF Report here)*

## Limitations
- Model predictions rely strictly on 2D slices.
- Unsuitable for multi-modal MRI (T1, T2, FLAIR) simultaneously unless explicitly re-trained.

## Medical Disclaimer
This system is intended for academic, research, and educational purposes only. It is not a certified medical diagnostic system and should not be used as a substitute for evaluation by a qualified medical professional.

## Future Scope
- 3D MRI Volume Support (NIfTI integration).
- Longitudinal tumor progression tracking.

## License
MIT License
"# DIF-SEG-Brain-Tumor" 
