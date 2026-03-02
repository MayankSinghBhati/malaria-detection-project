# Malaria Detection using Deep Learning (CNN)

## Overview

This project implements a Convolutional Neural Network (CNN) to detect malaria infection from microscopic blood cell images.

The model classifies images into two categories:
- Parasitized
- Uninfected

The system is built using TensorFlow and deployed using Streamlit for real-time prediction.

## Problem Statement

Traditional malaria diagnosis relies on microscopic examination of blood smear images. This process:

- Requires skilled pathologists
- Is time-consuming
- Can be inaccessible in remote areas

This project demonstrates how deep learning can assist in automating malaria detection from blood smear images.

## Dataset

- Dataset: NIH Malaria Cell Images
- Total Images: ~27,000
- Classes: 2 (Parasitized, Uninfected)
- Train-Validation Split: 80% Training, 20% Validation
- Image Size Used: 96 × 96

## Model Architecture

The CNN model consists of:

- Convolution Layers (feature extraction)
- ReLU Activation Function
- MaxPooling Layers
- Flatten Layer
- Dense Fully Connected Layer
- Sigmoid Output Layer (binary classification)

### Training Configuration

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epochs: 10
- Validation Accuracy: ~93%
- Training Accuracy: ~98%

## Results

- Validation Accuracy: ~93%
- Overall Evaluation Accuracy: ~97%
- Confusion Matrix and Classification Report generated

The small difference between training and validation accuracy indicates good generalization with minimal overfitting.

## Streamlit Application

The trained model is integrated into a Streamlit web application that allows users to:

- Upload a microscopic blood cell image
- Receive classification results
- View confidence scores for both classes

## Project Structure

MalariaProject/
- app.py
- train.py
- evaluate.py
- malaria_model.h5
- requirements.txt
- documentation/

## Installation & Setup

1. Install Anaconda.

2. Create and activate environment:

conda create -n malaria python=3.10  
conda activate malaria  

3. Install dependencies:

pip install -r requirements.txt  

4. Run the application:

streamlit run app.py  

Optional (Light Mode):

streamlit run app.py --theme.base="light"

## Limitations

- Not a clinical diagnostic tool
- Performance depends on image quality
- Misclassification rate approximately 7%
- Binary classification only

## Disclaimer

This project is developed for academic purposes only and is not intended for real-world medical diagnosis.
