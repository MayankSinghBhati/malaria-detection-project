🧬 Malaria Detection Using Deep Learning

An AI-powered web application that detects malaria parasites in microscopic blood cell images using Deep Learning and Transfer Learning.

This project compares multiple CNN architectures and deploys the best-performing model using Flask.

🚀 Project Overview

Malaria is a life-threatening disease caused by Plasmodium parasites transmitted through mosquito bites. Early detection through microscopic blood smear analysis is critical.

This project builds an automated malaria detection system using:

Custom Convolutional Neural Network (CNN)

MobileNetV2 (Transfer Learning)

EfficientNetB0 (Transfer Learning)

The final deployed model uses MobileNetV2 (Fine-Tuned) for optimal performance and generalization.

🧠 Model Comparison

Three architectures were trained and evaluated on the same dataset:

Model	Validation Accuracy
Custom CNN	~94%
MobileNetV2	~93%
EfficientNetB0	~50% (underperformed at 96×96 resolution)
Why MobileNetV2?

Although Custom CNN achieved slightly higher accuracy, MobileNetV2 was selected because:

Uses pretrained ImageNet weights

Better generalization capability

More scalable architecture

Transfer learning advantage

📊 Dataset

Microscopic blood smear cell images

Two classes:

Parasitized

Uninfected

Images resized to 96×96

Pixel values normalized to [0, 1]

🛠 Tech Stack

Python

TensorFlow / Keras

MobileNetV2 (Transfer Learning)

Flask

HTML / CSS

NumPy

PIL

🌐 Web Application Features

Upload microscopic blood cell image

Live image preview before prediction

AI-based classification

Confidence percentage

Class probability breakdown

Processing time display

Uploaded image preview in report

Clean dark-themed UI

Medical disclaimer included

🖥 Application Workflow

User uploads image.

Image is resized and normalized.

Model performs prediction.

Probabilities are computed.

Result page displays:

Prediction

Confidence level

Probability breakdown

Processing time

Uploaded image preview

📁 Project Structure
Malaria-Detection/
│
├── app.py
├── best_model.keras
├── train.py
├── model_comparison.py
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    └── uploads/
⚙️ How To Run Locally
1️⃣ Clone the Repository
git clone https://github.com/yourusername/malaria-detection.git
cd malaria-detection
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Application
python app.py

Open in browser:

http://127.0.0.1:5000
⚠️ Disclaimer

This AI system is developed for educational purposes only.
It is not intended for clinical or medical diagnosis.

📌 Future Improvements

Grad-CAM visualization for explainability

Batch image prediction

Drag-and-drop upload enhancement

Model performance dashboard

Deployment to cloud (Heroku / Render / AWS)

👨‍💻 Author

Developed as part of a Deep Learning project.

If you found this project useful, feel free to ⭐ the repository.
