import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# === Paths ===
YOLO_MODEL_PATH = "final_model.pt"
CLASSIFIER_PATH = "InceptionResNetV2_clothing_attribute_model.keras"
TEST_IMAGE_PATH = "dataset2/dress_0011.jpg"

# === Load Models ===
yolo_model = YOLO(YOLO_MODEL_PATH)
classifier = load_model(CLASSIFIER_PATH)

# === Class Names ===
CLASS_NAMES = ["Dress", "Outerwear", "Shirt", "Suit", "Sweater", "Tank Top", "T-shirt"]

# === Load and Prepare Image ===
image = cv2.imread(TEST_IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape

# === Run YOLO Detection ===
results = yolo_model(TEST_IMAGE_PATH)[0]

# === Make a Copy for Drawing ===
output_image = image_rgb.copy()

# === Process Detections ===
for box in results.boxes.data:
    x_min, y_min, x_max, y_max, conf, class_id = box.cpu().numpy()
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    # Crop and Resize for Classifier
    cropped_obj = image_rgb[y_min:y_max, x_min:x_max]
    cropped_resized = cv2.resize(cropped_obj, (299, 299)) / 255.0
    cropped_resized = np.expand_dims(cropped_resized, axis=0)

    # Predict with Custom Classifier
    class_probs = classifier.predict(cropped_resized)
    predicted_class = CLASS_NAMES[np.argmax(class_probs)]

    # Draw Box + Label
    cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    cv2.putText(output_image, predicted_class, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

# === Show Result ===
plt.figure(figsize=(8, 8))
plt.imshow(output_image)
plt.title("YOLO Detection + Our Classification Model")
plt.axis("off")
plt.show()
