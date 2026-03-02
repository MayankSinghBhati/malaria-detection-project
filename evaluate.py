import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

IMG_SIZE = 96

model = tf.keras.models.load_model("malaria_model.h5")

datagen = ImageDataGenerator(rescale=1./255)

val = datagen.flow_from_directory(
    "cell_images",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

predictions = model.predict(val)
predicted_classes = (predictions > 0.5).astype(int).reshape(-1)

print("Confusion Matrix:")
print(confusion_matrix(val.classes, predicted_classes))

print("\nClassification Report:")
print(classification_report(val.classes, predicted_classes))