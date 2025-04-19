import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIGURATION ---
MODEL_PATH = "resNet50_clothing_finetuned.keras"  # Update with your ResNet50 model path
IMG_SIZE = (224, 224)  # Standard size for ResNet50
BATCH_SIZE = 32
DATASET_PATH = "dataset2/crops"  # Update with your dataset path

# --- Category Dictionary ---
category_dict = {
    0: 'Dress',
    1: 'Outerwear',
    2: 'Shirt',
    3: 'Suit',
    4: 'Sweater',
    5: 'Tank Top',
    6: 'T-shirt'
}

print("[DEBUG] Loading model from:", MODEL_PATH)
# --- Load the Saved Model ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- Data Generator for Evaluation ---
test_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# Use the validation subset as the test set; shuffle=False for consistent order
test_ds = test_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# --- Get True Labels ---
true_labels = test_ds.classes
class_indices = test_ds.class_indices
num_classes = len(class_indices)
print(f"[DEBUG] Number of classes detected: {num_classes}")

# --- Predictions ---
print("[DEBUG] Predicting on test dataset...")
predictions = model.predict(test_ds)
pred_labels = np.argmax(predictions, axis=1)

# --- Calculate Metrics ---
accuracy_val = accuracy_score(true_labels, pred_labels)
precision_val = precision_score(true_labels, pred_labels, average=None)
recall_val = recall_score(true_labels, pred_labels, average=None)
f1_val = f1_score(true_labels, pred_labels, average=None)

# --- Confusion Matrix ---
conf_matrix = confusion_matrix(true_labels, pred_labels)

# --- Generate Classification Report (as dictionary) ---
report_dict = classification_report(
    true_labels, pred_labels, target_names=list(category_dict.values()), output_dict=True
)

# --- Print Detailed Report per Label (in percentages) ---
print("\nDetailed Classification Report (per label):")
for label in list(category_dict.values()):
    metrics = report_dict[label]
    print(f"Label: {label}")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {metrics['f1-score']*100:.2f}%")
    print(f"  Support:   {int(metrics['support'])}")
    print()

# Print overall metrics
print("Overall Accuracy: {:.2f}%".format(report_dict["accuracy"]*100))
print("Macro Average Precision: {:.2f}%".format(report_dict["macro avg"]["precision"]*100))
print("Macro Average Recall:    {:.2f}%".format(report_dict["macro avg"]["recall"]*100))
print("Macro Average F1-Score:  {:.2f}%".format(report_dict["macro avg"]["f1-score"]*100))
print("Weighted Average Precision: {:.2f}%".format(report_dict["weighted avg"]["precision"]*100))
print("Weighted Average Recall:    {:.2f}%".format(report_dict["weighted avg"]["recall"]*100))
print("Weighted Average F1-Score:  {:.2f}%".format(report_dict["weighted avg"]["f1-score"]*100))

# --- Plot Confusion Matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(category_dict.values()),
            yticklabels=list(category_dict.values()))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# --- Prepare Data for Bar Graph (Percentages) ---
metrics_data = {
    "Labeling Rate": [accuracy_val * 100] * num_classes,
    "Precision": (precision_val * 100).tolist(),
    "Recall": (recall_val * 100).tolist()
}

labels = list(category_dict.values())
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, metrics_data["Labeling Rate"], width, label="Labeling Rate", color='blue')
ax.bar(x, metrics_data["Precision"], width, label="Precision", color='orange')
ax.bar(x + width, metrics_data["Recall"], width, label="Recall", color='green')

ax.set_xlabel("Clothing Attributes")
ax.set_ylabel("Percentage (%)")
ax.set_title("Clothing Attribute Recognition Performance")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()

# Show Bar Graph
plt.show()
