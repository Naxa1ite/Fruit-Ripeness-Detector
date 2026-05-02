# 🍎 Fruit Ripeness Detection Using Deep Learning

> A CNN-based web application that detects the ripeness stage of fruits from photos and estimates days until the next ripeness transition.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat&logo=flask&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-CNN-D00000?style=flat&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 Overview

This project uses a **Convolutional Neural Network (CNN)** trained on images of **apples, bananas, and oranges** to:

- Classify the **ripeness stage** of a fruit — `Unripe`, `Ripe`, or `Rotten`
- Estimate the **number of days** until the fruit ripens or goes rotten
- Serve predictions through a **Flask REST API** with a clean web UI

Built as a Deep Learning course project at **Vellore Institute of Technology, Chennai**.

---

## 🎯 Features

- 📷 **Drag & drop image upload** — no app installation needed
- 🧠 **CNN inference** — real-time ripeness classification
- 📅 **Days estimation** — confidence-based shelf-life countdown
- 📊 **Confidence bars** — visual breakdown of all three class probabilities
- 💡 **Storage tips** — contextual advice based on ripeness result
- 🌐 **REST API** — `/predict` endpoint returns structured JSON

---

## 🍌 Supported Fruits & Classes

| Fruit  | Unripe | Ripe | Rotten |
|--------|--------|------|--------|
| 🍎 Apple  | ✅ | ✅ | ✅ |
| 🍌 Banana | ✅ | ✅ | ✅ |
| 🍊 Orange | ✅ | ✅ | ✅ |

---

## 🧠 Model Architecture

```
Input (224×224×3)
    → Conv2D(32, 3×3, ReLU)
    → MaxPooling2D(2×2)
    → Conv2D(64, 3×3, ReLU)
    → MaxPooling2D(2×2)
    → Flatten
    → Dense(128, ReLU)
    → Dense(3, Softmax)   ← outputs: [ripe, rotten, unripe]
```

- **Loss:** Categorical Cross-Entropy  
- **Optimiser:** Adam (lr = 0.001)  
- **Input size:** 224 × 224 px  
- **Test Accuracy:** ~91% weighted average F1

---

## 📅 Days Estimation Logic

```python
p_ripe   = probs[classes.index('ripe')]
p_rotten = probs[classes.index('rotten')]

if stage == "unripe":
    days = int((1 - p_ripe) * 5) + 1
    msg  = f"{days} days to become ripe"

elif stage == "ripe":
    days = int((1 - p_rotten) * 3) + 1
    msg  = f"{days} days to become rotten"

else:
    msg = "Already rotten"
```

---

## 🗂️ Project Structure

```
fruit-ripeness-detection/
├── app.py                   # Flask backend & prediction API
├── best_fruit_model.h5      # Trained Keras model (not tracked by git)
├── templates/
│   └── index.html           # Frontend UI
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fruit-ripeness-detection.git
cd fruit-ripeness-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the model file

Place your trained `best_fruit_model.h5` in the project root directory.

### 4. Run the app

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## 📦 Requirements

```
flask
tensorflow
pillow
numpy
```

Or install via:

```bash
pip install flask tensorflow pillow numpy
```

---

## 🔌 API Reference

### `POST /predict`

Upload a fruit image and get a ripeness prediction.

**Request:** `multipart/form-data` with field `file` (image)

**Response:**
```json
{
  "ripeness": "ripe",
  "confidence": {
    "ripe": 0.9412,
    "rotten": 0.0381,
    "unripe": 0.0207
  },
  "days": {
    "days": 2,
    "message": "2 days to become rotten"
  }
}
```

### `GET /health`

Returns model status and loaded class names.

---

## 📊 Results

| Class    | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Unripe   | 0.93      | 0.91   | 0.92     |
| Ripe     | 0.92      | 0.94   | 0.93     |
| Rotten   | 0.90      | 0.89   | 0.89     |
| **Avg**  | **0.92**  | **0.91** | **0.91** |

| Fruit  | Test Accuracy |
|--------|--------------|
| Apple  | 92.4%        |
| Banana | 94.1%        |
| Orange | 89.7%        |

---

## 👥 Team

| Name | Reg. No. |
|------|----------|
| Ayan Chandrakar | 23BAI1522 |
| Om Shukla | 23BAI1488 |
| Abhimanyu Sood | 23BAI1352 |

**Vellore Institute of Technology, Chennai**  
Deep Learning — Academic Year 2025–26

---

## 📄 License

This project is licensed under the MIT License.
