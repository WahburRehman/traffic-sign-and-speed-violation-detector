# 🚦 Traffic Sign & Speed Violation Detector

A Computer Vision project that detects **traffic signs** in real-time videos and raises **speed violation alerts** using a YOLOv8-based deep learning model.  
Built with **Python, YOLOv8, PyTorch, OpenCV, and Streamlit** — this project demonstrates how AI can be applied in **intelligent transportation systems**.

---

## 📌 Features
- Detects **43 classes** of German traffic signs (speed limits, prohibitions, warnings, etc.)
- **Speed violation detection** (customizable speed slider in Streamlit UI)
- Real-time **video processing**
- Clean and simple **Streamlit Web App**

---

## 📊 Demo
👉 Try the live Streamlit app here:  
[Streamlit App](https://traffic-sign-and-speed-violation-detector-fseayw3pwclcsxf3tivu.streamlit.app/)

---

## 📂 Dataset
This project is trained on a **custom YOLO-format dataset** based on:

- **GTSDB (German Traffic Sign Detection Benchmark)**  
- **GTSRB (German Traffic Sign Recognition Benchmark)**  
- Additional **synthetic augmentation** (background pasting, rare-class balancing)

🔗 Download the dataset on Kaggle:  
[German Traffic Sign Detection Dataset (YOLO, Aug + Org)](https://www.kaggle.com/datasets/wahburrehman/german-traffic-signs-detection-yolo-aug-org)

---

## 🧠 Model Weights
The YOLOv8 model is trained on the above dataset with augmentation.  
You can directly download the trained model weights here:
🔗 [Hugging Face Model Repo](https://huggingface.co/WahburRehman/traffic-sign-detector/tree/main)

---

## 🏗️ Project Architecture
1. **Data Preparation**  
   - Merge & preprocess GTSRB + GTSDB  
   - Augment rare classes with synthetic placement  
   - Convert to YOLOv8 format (`images/`, `labels/`, `data.yaml`)

2. **Model Training**  
   - YOLOv8 (Ultralytics) fine-tuned on the dataset  
   - Early stopping + augmentation ratio schedule

3. **Streamlit Application**  
   - Upload a video or choose a sample  
   - Detection overlay with bounding boxes  
   - Violation logic (speed slider + alerts)  
   - Results preview & optional CSV export

---

## ⚙️ Installation
**Clone the repository:**
```bash
git clone https://github.com/WahburRehman/traffic-sign-and-speed-violation-detector.git
cd traffic-sign-violation-detector
```
**Create and activate environment:**
```bash
conda create -n traffic_sign_detection python=3.10 -y
conda activate traffic_sign_detection
```
**Install dependencies:**
```bash
pip install -r requirements.txt
```
---

## 🚀 Usage
```bash
streamlit run app.py
```
---

## 🙌 Acknowledgements
- Ultralytics YOLOv8
- GTSRB Dataset
- GTSDB Dataset
- Kaggle & Hugging Face community
