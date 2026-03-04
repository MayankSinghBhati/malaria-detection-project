import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os

IMG_SIZE = 96
BATCH_SIZE = 32
DATASET_PATH = "cell_images"

# Load validation data
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

true_labels = val_generator.classes

# Load BEST MobileNetV2 model
model = tf.keras.models.load_model("best_model.h5")

# Predict probabilities
predictions = model.predict(val_generator)
predicted_classes = (predictions > 0.5).astype(int).reshape(-1)

print("\n========== MobileNetV2 Evaluation ==========")

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predicted_classes))

print("\nClassification Report:")
print(classification_report(true_labels, predicted_classes))

# ROC Curve
fpr, tpr, thresholds = roc_curve(true_labels, predictions)
roc_auc = auc(fpr, tpr)

print(f"\nROC-AUC Score: {roc_auc:.4f}")

plt.figure()
plt.plot(fpr, tpr, label=f'MobileNetV2 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - MobileNetV2')
plt.legend(loc="lower right")
plt.grid()

os.makedirs("../documentation", exist_ok=True)

plt.savefig("../documentation/MobileNetV2_ROC.png")
plt.show()