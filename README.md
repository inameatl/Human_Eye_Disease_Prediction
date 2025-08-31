# OCT Retinal Analysis Platform

## Overview
This project is a **Retinal OCT (Optical Coherence Tomography) Analysis Platform** built with **Streamlit** and **TensorFlow**. It allows users to **upload OCT images** and automatically classify them into **CNV, DME, Drusen, or Normal**. The platform also provides detailed insights and recommendations for each retinal condition.

---

## Features
- **Automated Image Classification**: Detects CNV, DME, Drusen, and Normal retina.
- **Interactive Dashboard**: Built with Streamlit for easy use.
- **Detailed Recommendations**: Provides information and guidance based on predicted disease.
- **High Accuracy**: Trained on a verified dataset of **84,495 OCT images**.

---

## Dataset
The dataset contains images from multiple sources, verified by ophthalmologists and retinal specialists:

- Shiley Eye Institute
- California Retinal Research Foundation
- Medical Center Ophthalmology Associates
- Shanghai First People’s Hospital
- Beijing Tongren Eye Center

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/inameatl/Human_Eye_Disease_Prediction.git
cd Human_Eye_Disease_Prediction

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt

streamlit run app.py
