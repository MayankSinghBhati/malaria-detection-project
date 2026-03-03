# 🧬 Malaria Detection using Deep Learning (CNN + Transfer Learning)
## 📌 Overview

This project implements a deep learning system for detecting malaria infection from microscopic blood smear images.

The system performs binary classification:

Parasitized

Uninfected

Multiple CNN architectures were trained, evaluated, and compared.

Two architectures were developed:

CustomCNN (Lightweight) – Used for demonstration and lightweight deployment

MobileNetV2 (Transfer Learning) – Final production architecture aligned with milestone requirements

The final production architecture follows the milestone flow using MobileNetV2 with fine-tuning.

## 🎯 Problem Statement

Manual malaria diagnosis:

Requires trained experts

Is time-consuming

Can be inaccessible in remote areas

Is prone to human error

This project demonstrates how Convolutional Neural Networks (CNNs) can assist in automated malaria screening.

## 📂 Dataset

Dataset: NIH Malaria Cell Images
Total Images: ~27,000
Classes: 2 (Parasitized, Uninfected)
Train–Validation Split: 80% Training, 20% Validation
Image Resolution Used: 96 × 96 pixels

https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

**🔹 Data Preprocessing**

Images resized to 96×96

Pixel normalization to range [0,1]

Train–Validation split (80–20)

Data generators used for efficient training

## 🧠 Model Development
**1️⃣ CustomCNN (Lightweight Deployment Model)**

Validation Accuracy: ~94%

Purpose:

Lightweight deployment

Faster inference

Demonstration model for web interface

**2️⃣ MobileNetV2 (Transfer Learning – Milestone Architecture)**

Validation Accuracy: ~93%

Configuration:

Pretrained on ImageNet

include_top=False

Fine-tuned final layers

GlobalAveragePooling

Dense classification head

## ⚙️ Training Configuration

Framework: TensorFlow 2.13.0
Keras Version: 2.13.x
Python Version: 3.10

Optimizer: Adam
Loss Function: Binary Crossentropy
Epochs: 10
Callbacks Used:

EarlyStopping

ReduceLROnPlateau

ModelCheckpoint

Training Accuracy: ~98%
Validation Accuracy: ~93–94%

Minimal overfitting observed.

## 🌐 Deployment

The application is deployed using:

HuggingFace Spaces

Gradio UI

TensorFlow 2.13 runtime

**🔹 Demo Model**

CustomCNN is used in lightweight demo environments for faster inference.

**🔹 Production Architecture**

MobileNetV2 (Transfer Learning) represents the final milestone-compliant architecture.

This distinction ensures:

Milestone alignment

Demonstration efficiency

Architectural clarity

## ✨ Features

Upload microscopic blood cell image

Real-time prediction

Confidence score display

Processing time measurement

Professional medical-style UI

Model variant transparency

Academic disclaimer

## 🔄 Application Workflow

User uploads a blood smear image

Image is resized and normalized

CNN performs binary classification

Prediction probability calculated

Results displayed with confidence and processing time

## 📁 Project Structure
Malaria_Detection_Project/
│
├── static
├── templates
├── app.py
├── train.py
├── model_comparison.py
├── visualize_data.py
├── evaluate.py
├── best_model.h5
├── malaria_model.h5
├── requirements.txt
├── runtime.txt
└── README.md
## 📊 Model Comparison Summary
Model	Validation Accuracy	Use Case
CustomCNN	~94%	Demo / Lightweight
MobileNetV2	~93%	Production / Milestone

Both models demonstrate strong performance with minimal overfitting.

## ⚠️ Limitations

Not a clinical diagnostic tool

Dependent on image quality

Binary classification only

Resolution constrained to 96×96

## 📜 Disclaimer

**This project is developed strictly for academic and educational purposes.**
**It is not intended for medical diagnosis or clinical use.**
