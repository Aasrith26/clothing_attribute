import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "VGG16_clothing_attribute_model_final_1.keras"  # Update if needed
IMG_SIZE = (224, 224)   # Matches VGG16 input size
BATCH_SIZE = 32
DATASET_PATH = "dataset2/val"  # Folder containing subfolders per class

# Define your classes here:
category_dict = {
    0: "Dress",
    1: "Outerwear",
    2: "Shirt",
    3: "Suit",
    4: "Sweater",
    5: "Tank Top",
    6: "T-shirt"
}

# -----------------------------
# Load Model
# -----------------------------
print("[DEBUG] Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Data Generator
# -----------------------------
# Using ImageDataGenerator with rescale ONLY, no validation split
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Flow from directory - uses entire folder as test/validation set
test_ds = test_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # for consistent order
)

# -----------------------------
# Predictions
# -----------------------------
print("[DEBUG] Predicting on dataset...")
predictions = model.predict(test_ds)
pred_labels = np.argmax(predictions, axis=1)

# -----------------------------
# True Labels & Class Indices
# -----------------------------
true_labels = test_ds.classes
class_indices = test_ds.class_indices
num_classes = len(class_indices)
print(f"[DEBUG] Number of classes detected: {num_classes}")
print("[DEBUG] class_indices:", class_indices)

# -----------------------------
# Calculate Metrics
# -----------------------------
accuracy_val = accuracy_score(true_labels, pred_labels)
precision_val = precision_score(true_labels, pred_labels, average=None)
recall_val    = recall_score(true_labels, pred_labels, average=None)
f1_val        = f1_score(true_labels, pred_labels, average=None)

conf_matrix   = confusion_matrix(true_labels, pred_labels)
report_dict   = classification_report(
    true_labels,
    pred_labels,
    target_names=list(category_dict.values()),
    output_dict=True
)

# -----------------------------
# Print Detailed Report
# -----------------------------
print("\nDetailed Classification Report (per label):")
for label_name in list(category_dict.values()):
    metrics = report_dict[label_name]
    print(f"Label: {label_name}")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['f1-score']*100:.2f}%")
    print(f"  Support:   {int(metrics['support'])}\n")

# Overall metrics
print("Overall Accuracy: {:.2f}%".format(report_dict["accuracy"] * 100))
print("Macro Avg Precision: {:.2f}%".format(report_dict["macro avg"]["precision"] * 100))
print("Macro Avg Recall:    {:.2f}%".format(report_dict["macro avg"]["recall"] * 100))
print("Macro Avg F1-Score:  {:.2f}%".format(report_dict["macro avg"]["f1-score"] * 100))
print("Weighted Avg Precision: {:.2f}%".format(report_dict["weighted avg"]["precision"] * 100))
print("Weighted Avg Recall:    {:.2f}%".format(report_dict["weighted avg"]["recall"] * 100))
print("Weighted Avg F1-Score:  {:.2f}%".format(report_dict["weighted avg"]["f1-score"] * 100))

# -----------------------------
# Confusion Matrix Visualization
# -----------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(category_dict.values()),
            yticklabels=list(category_dict.values()))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("VGG16 Model - Confusion Matrix")
plt.show()

# -----------------------------
# Bar Graph Visualization
# -----------------------------
import numpy as np
x = np.arange(num_classes)
width = 0.25

metrics_data = {
    "Labeling Rate": [accuracy_val * 100] * num_classes,
    "Precision": (precision_val * 100).tolist(),
    "Recall": (recall_val * 100).tolist(),
}

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, metrics_data["Labeling Rate"], width, label="Labeling Rate", color='blue')
ax.bar(x, metrics_data["Precision"], width, label="Precision", color='orange')
ax.bar(x + width, metrics_data["Recall"], width, label="Recall", color='green')

ax.set_xlabel("Clothing Classes")
ax.set_ylabel("Percentage (%)")
ax.set_title("Clothing Attribute Recognition Performance (VGG16)")
ax.set_xticks(x)
ax.set_xticklabels(list(category_dict.values()), rotation=45, ha="right")
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.show()
