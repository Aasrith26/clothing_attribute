import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# -----------------------------
# Configuration
# -----------------------------
YOLO_MODEL_PATH = "final_model.pt"  # Your YOLO model path
CLASSIFIER_PATH = "VGG16_clothing_attribute_model.keras"  # Trained VGG16 model
TEST_IMAGE_PATH = "dataset2/dress_0011.jpg"

# VGG16 expects (224, 224)
IMG_SIZE = (224, 224)

# Class names
CLASS_NAMES = [
    "Dress", "Outerwear", "Shirt",
    "Suit", "Sweater", "Tank Top", "T-shirt"
]

# -----------------------------
# Load Models
# -----------------------------
print("[DEBUG] Loading YOLO model from:", YOLO_MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)

print("[DEBUG] Loading VGG16 classifier from:", CLASSIFIER_PATH)
classifier = load_model(CLASSIFIER_PATH)

# -----------------------------
# Load and Prepare Image
# -----------------------------
image = cv2.imread(TEST_IMAGE_PATH)
if image is None:
    raise ValueError(f"Unable to load image: {TEST_IMAGE_PATH}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape


results = yolo_model(TEST_IMAGE_PATH)[0]

# Make a copy for drawing
output_image = image_rgb.copy()


for box in results.boxes.data:
    x_min, y_min, x_max, y_max, conf, class_id = box.cpu().numpy()
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    # Crop and Resize for VGG16
    cropped_obj = image_rgb[y_min:y_max, x_min:x_max]
    cropped_resized = cv2.resize(cropped_obj, IMG_SIZE) / 255.0
    cropped_resized = np.expand_dims(cropped_resized, axis=0)

    # Predict with VGG16
    class_probs = classifier.predict(cropped_resized)
    predicted_class = CLASS_NAMES[np.argmax(class_probs)]

    # Draw bounding box + label
    cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    cv2.putText(output_image, predicted_class, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

# -----------------------------
# Show Result
# -----------------------------
plt.figure(figsize=(8, 8))
plt.imshow(output_image)
plt.title("YOLO Detection + VGG16 Classification")
plt.axis("off")
plt.show()
