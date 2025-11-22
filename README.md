#🚦 Traffic Sign Recognition System
An end-to-end deep learning project that recognizes 43 types of traffic signs using a Convolutional Neural Network (CNN) built with TensorFlow/Keras, combined with a Streamlit web app for real-time predictions.

#✨ Overview
This project implements a complete machine learning pipeline to classify German Traffic Sign Recognition Benchmark (GTSRB) images.
It includes:
Data preprocessing
Exploratory Data Analysis (EDA)
CNN model training
Evaluation & metrics
Real-time prediction interface via Streamlit

The trained model achieves:
👉 98.72% Test Accuracy 🎉

#📂 Project Structure
traffic-sign-recognition/
│
├── data/
│   ├── Train/         # Training images (43 classes)
│   ├── Test/          # Test images
│   ├── Meta/          # Metadata files
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── streamlit_app.py
│
├── models/
│   ├── best_model.h5   # Saved trained model
│
├── outputs/
│   ├── plots/          # Accuracy/loss curves, confusion matrix, etc.
│
├── main.py
├── requirements.txt
└── README.md

#📊 Dataset — GTSRB
The project uses the German Traffic Sign Recognition Benchmark, containing:
50,000+ images
43 classes
Real-world variations: brightness, noise, rotation, occlusion
Each image is resized to 30×30×3 before training.

#🧠 Model Architecture (CNN)
The CNN consists of:
Convolution layers (ReLU + BatchNorm)
MaxPooling layers
Dropout regularization
Fully connected dense layers
Softmax output layer (43 classes)
The model is optimized using:
Adam optimizer (lr=0.001)
Batch Size: 64
Epochs: 20
Categorical Cross-Entropy Loss

#📈 Training & Evaluation
After preprocessing and augmentation, the final model achieved:
Metric	Value
Test Accuracy	98.72%
Test Loss	0.0549
Epochs	20
Batch Size	64

Additional evaluation outputs include:
Confusion Matrix
Accuracy Curve
Loss Curve
Classification Report

#⚙️ How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Activate virtual environment (optional but recommended)
python -m venv new
new\Scripts\activate

3️⃣ Run each pipeline step
👉 Preprocess data
python main.py --data

👉 Exploratory Data Analysis
python main.py --eda

👉 Train the model
python main.py --training

👉 Evaluate the model
python main.py --evaluation

👉 Run inference on an image
python main.py --inference --image_path "path/to/image.png"

#🌐 Run the Streamlit Web App
Start the UI where users can upload images and get predictions:
streamlit run scripts/streamlit_app.py

This launches the app at:
👉 http://localhost:8501
Upload a traffic sign image to see real-time predictions.

#📝 Credits
Dataset: German Traffic Sign Recognition Benchmark (GTSRB)
Frameworks: TensorFlow, Keras, Streamlit, scikit-learn
Developed by: Shruti Khandelwal
