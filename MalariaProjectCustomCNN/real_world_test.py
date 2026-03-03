import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import time
import random

model = load_model("best_model.h5")

folder = "../test_images/"
total = 0
correct = 0
start = time.time()

count = 0

for label in ["Parasitized", "Uninfected"]:
    path = os.path.join(folder, label)

    all_images = [img for img in os.listdir(path)
                  if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    selected_images = random.sample(all_images, min(100, len(all_images)))

    for img_name in selected_images:
        count += 1
        print(f"Processing {count}: {img_name}")

        img_path = os.path.join(path, img_name)

        img = Image.open(img_path).resize((96, 96))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)

        predicted_label = "Uninfected" if pred[0][0] > 0.5 else "Parasitized"

        if predicted_label == label:
            correct += 1
        total += 1

end = time.time()

print("Batch Accuracy:", correct/total)
print("Average Inference Time:", (end-start)/total)