🩺 **Pneumonia Detection from Chest X-Ray Images using EfficientNetB0**

A deep learning model to classify **Chest X-Ray images** into **Pneumonia** or **Normal** using **EfficientNetB0 (Transfer Learning)**.
This project demonstrates dataset preprocessing, model training, evaluation metrics, visualization, and medical imaging analysis.

---

📌 **Project Overview**

Pneumonia is one of the leading causes of hospitalizations.
Chest X-rays are the primary diagnostic tool, but manual interpretation can be challenging.

This project uses deep learning to **automatically detect pneumonia** from X-ray images with high accuracy.

---
 📂 **Dataset**

**Source:**
Kaggle — *Chest X-Ray Images (Pneumonia)*
[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Classes:**

* NORMAL
* PNEUMONIA

**Dataset contains:**

* Training images (used with validation split)
* Separate test set (624 images)

> The dataset is not included in this repo due to size restrictions.
> Please download it from Kaggle and place it inside `data/`.

---

🧠 **Model Architecture**

### ✔ EfficientNetB0 (pretrained on ImageNet)

* Frozen convolutional base
* Preprocessing layers included
* GlobalAveragePooling2D
* Dropout (0.3)
* Fully connected layer → Softmax (2 classes)

This gives strong performance with less training time.

---

🔧 **Technologies Used**

| Purpose       | Library             |
| ------------- | ------------------- |
| Deep Learning | TensorFlow / Keras  |
| Data Pipeline | tf.data             |
| Metrics       | Scikit-Learn        |
| Visualization | Matplotlib, Seaborn |

---

 🚀 **How to Run**

### 1️⃣ Install requirements

pip install -r requirements.txt

### 2️⃣ Download and place the dataset

data/chest_xray/train/
data/chest_xray/test/


### 3️⃣ Open and run the notebook

Chest_Xray_Pneumonia_EfficientNet.ipynb

---

 📊 **Results**

### ✔ **Training & Validation Accuracy**

(Insert your `accuracy.png`)

### ✔ **Training & Validation Loss**

(Insert your `loss.png`)

### ✔ **Confusion Matrix**

Shows classification performance on each class.
(Insert your `confusion_matrix.png`)

### ✔ **Classification Report**

Includes precision, recall, F1-score.

### ✔ **Sample Predictions**

Displays random test images with predictions.

### ✔ **ROC Curve**

AUC score demonstrates performance beyond simple accuracy.
(Insert `roc_curve.png`)

---

## 📌 **Folder Structure**

project/
│
├── notebooks/
│   └── Chest_Xray_Pneumonia_EfficientNet.ipynb
│
├── results/
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── sample_predictions.png
│
├── saved_models/
│   └── best_model.h5
│
├── requirements.txt
└── README.md


## 👤 **Author**

**Surabhi H R**

M.Sc Data Science



