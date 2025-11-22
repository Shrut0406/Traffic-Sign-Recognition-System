# 🚦 Traffic Sign Recognition System

An end-to-end deep learning project that recognizes **43 types of traffic signs** using a Convolutional Neural Network (CNN) built with **TensorFlow/Keras**, combined with a **Streamlit web app** for real-time predictions.

---

## ✨ Overview

This project implements a complete machine learning pipeline to classify German Traffic Sign Recognition Benchmark (**GTSRB**) images.

It includes:

- Data preprocessing  
- Exploratory Data Analysis (EDA)  
- CNN model training  
- Evaluation & metrics  
- Real-time prediction interface via Streamlit  

The trained model achieves:

👉 **98.72% Test Accuracy 🎉**

---

## 📂 Project Structure

traffic-sign-recognition/
│
├── data/
│ ├── Train/ # Training images (43 classes)
│ ├── Test/ # Test images
│ ├── Meta/ # Metadata files
│
├── scripts/
│ ├── data_preprocessing.py
│ ├── eda.py
│ ├── model_training.py
│ ├── evaluation.py
│ ├── inference.py
│ ├── streamlit_app.py
│
├── models/
│ ├── best_model.h5 # Saved trained model
│
├── outputs/
│ ├── plots/ # Accuracy/loss curves, confusion matrix, etc.
│
├── main.py
├── requirements.txt
└── README.md


---

## 📊 Dataset — GTSRB

The **German Traffic Sign Recognition Benchmark** dataset includes:

- **50,000+ images**  
- **43 traffic sign categories**  
- Real-world variations such as noise, blur, rotation, lighting differences, and occlusion  

All images are resized to **30×30×3** before training.

---

## 🧠 Model Architecture (CNN)

The CNN includes:

- Convolutional layers (ReLU + BatchNormalization)  
- MaxPooling layers  
- Dropout regularization  
- Dense (fully connected) layers  
- Softmax output layer (43 classes)  

**Training configuration:**

- **Optimizer:** Adam (learning rate = 0.001)  
- **Batch size:** 64  
- **Epochs:** 20  
- **Loss function:** Categorical Crossentropy  

---

## 📈 Training & Evaluation

**Final model performance:**

| Metric         | Value      |
|----------------|------------|
| Test Accuracy  | **98.72%** |
| Test Loss      | 0.0549     |
| Epochs         | 20         |
| Batch Size     | 64         |

Additional evaluation includes:

- Confusion Matrix  
- Accuracy Curve  
- Loss Curve  
- Classification Report  

---

## ⚙️ How to Run the Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt

2️⃣ (Optional) Create & activate a virtual environment
python -m venv new
new\Scripts\activate

3️⃣ Run pipeline components
➤ Preprocess data
python main.py --data

➤ Exploratory Data Analysis
python main.py --eda

➤ Train the model
python main.py --training

➤ Evaluate the model
python main.py --evaluation

➤ Run inference on a single image
python main.py --inference --image_path "path/to/image.png"

🌐 Run the Streamlit Web App

Start the interactive UI:

streamlit run scripts/streamlit_app.py


After launching, visit:

👉 http://localhost:8501

Upload a traffic sign image to get real-time predictions.

📝 Credits

Dataset: GTSRB — German Traffic Sign Recognition Benchmark

Frameworks: TensorFlow, Keras, Streamlit, scikit-learn

Developed by: Shruti Khandelwal

