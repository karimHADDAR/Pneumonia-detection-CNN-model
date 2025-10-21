# 🩻 Pneumonia Detection using Convolutional Neural Networks (CNN)

A deep learning project for detecting **Pneumonia** from chest X-ray images using a **DenseNet121**-based Convolutional Neural Network.  
The project includes model training, fine-tuning, Grad-CAM visualization for interpretability, and a **Streamlit web app** for easy deployment and testing.

---

## 🚀 Features

- 🧠 **Transfer Learning with DenseNet121**
- 🩺 **Binary Classification:** Normal vs. Pneumonia
- 🌈 **Grad-CAM Heatmaps** for explainable AI
- 💻 **Streamlit Web App** for user-friendly predictions
- 📈 **Evaluation Metrics:** Accuracy and ROC-AUC

---

## 🧩 Project Structure

pneumonia-cnn/
│
├── app.py # Streamlit web app for prediction and Grad-CAM
├── train.py # Training script for DenseNet121
│
├── src/
│ └── utils.py # Utility functions (Grad-CAM, image preprocessing, etc.)
│
├── data/ # Dataset directory (train/val/test)
│
├── runs/ # Model outputs (e.g., pneumonia_densenet121.h5)
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/karimHADDAR/Pneumonia-detection-CNN-model.git
cd Pneumonia-detection-CNN-model
2️⃣ Create a Virtual Environment
bash
Copy code
python -m venv .venv
.\.venv\Scripts\activate
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
🏋️ Train the Model
Make sure your dataset is organized as:

kotlin
Copy code
data/
 ├── train/
 │    ├── NORMAL/
 │    └── PNEUMONIA/
 ├── val/
 └── test/
Then run:

bash
Copy code
python train.py
This will:

Train the DenseNet121-based model

Fine-tune it on the validation data

Save the trained model to runs/pneumonia_densenet121.h5

🌐 Run the Streamlit App
Once the model is trained (or downloaded), run the Streamlit web app:

bash
Copy code
streamlit run app.py
Open your browser and go to:
👉 http://localhost:8501

You’ll see a web interface to upload chest X-ray images and visualize:

Prediction (Normal / Pneumonia)

Grad-CAM attention heatmap

🧠 Model Details
The model uses DenseNet121 pretrained on ImageNet for feature extraction, with a custom classification head.

python
Copy code
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])
We then fine-tune the entire model with a smaller learning rate to improve accuracy.

🩺 Explainability (Grad-CAM)
The project includes Grad-CAM visualization to show which lung regions the model focused on.
This helps doctors interpret predictions and verify that the model is learning meaningful features.

Example:

Input X-ray	Grad-CAM Heatmap

📊 Evaluation
Metric	Value
Validation Accuracy	~92%
Test ROC-AUC	~0.95

(Values may vary depending on training configuration)

🧾 Requirements
Main libraries used:

TensorFlow / Keras

NumPy

scikit-learn

OpenCV

Matplotlib

Streamlit

Install all with:

bash
Copy code
pip install -r requirements.txt
📦 Model Download
You can download the pretrained model from external storage:
👉 Google Drive Link

Save it to:

bash
Copy code
runs/pneumonia_densenet121.h5

🧑‍💻 Author

Karim Haddar
📧 karimhaddar24@example.com

🔗 GitHub 


🔗 LinkedIn



⭐ Acknowledgements

Dataset: Chest X-Ray Images (Pneumonia) - Kaggle

Pretrained model: DenseNet121 (ImageNet)
