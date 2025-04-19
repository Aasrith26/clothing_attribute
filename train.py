import os
import pickle
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
from collections import Counter
from tensorflow.keras.callbacks import ReduceLROnPlateau

from config import *
from feature_extraction import get_feature_extractor
from classification_model import build_classification_model, compile_classification_model
from boundary_correction import build_boundary_correction_model, compile_boundary_correction_model

# === Load Precomputed Data === #
PRECOMPUTED_FILE = "training_samples_restnet50.pkl"

if not os.path.exists(PRECOMPUTED_FILE):
    raise FileNotFoundError(f"{PRECOMPUTED_FILE} not found. Run precompute_regions.py first.")

with open(PRECOMPUTED_FILE, "rb") as f:
    data = pickle.load(f)

all_samples = data["samples"]
print(f"[INFO] Loaded {len(all_samples)} samples from {PRECOMPUTED_FILE}.")

# === Debugging: Print First Sample === #
print(f"[DEBUG] First Sample Structure: {all_samples[0]}")

# === Count Frequency of Categories === #
category_counts = Counter([s[2] for s in all_samples])
print("\n[DEBUG] Category Frequency Distribution BEFORE Filtering:")
for cat, count in category_counts.items():
    print(f"  Category {cat}: {count}")

# === Separate Positive and Background Samples === #
positive_samples = [s for s in all_samples if s[2] > 0]  # Clothing
background_samples = [s for s in all_samples if s[2] == 0]  # Background

# === Enforce 70:30 Ratio (Background : Positive) === #
num_pos = len(positive_samples)
num_neg_to_keep = int(num_pos * 0.3 / 0.7)  # Reduce background sample percentage to 30%
filtered_background_samples = background_samples[:num_neg_to_keep]

# Merge and Shuffle Dataset
filtered_samples = positive_samples + filtered_background_samples
np.random.shuffle(filtered_samples)

# === Debugging: Show Category Distribution After Filtering === #
final_category_counts = Counter([s[2] for s in filtered_samples])
print("\n[DEBUG] Category Frequency Distribution AFTER Filtering:")
for cat, count in final_category_counts.items():
    print(f"  Category {cat}: {count}")

# === Train/Val Split === #
split_idx = int(0.8 * len(filtered_samples))
train_samples = filtered_samples[:split_idx]
val_samples = filtered_samples[split_idx:]
print(f"[INFO] Train set: {len(train_samples)}, Val set: {len(val_samples)}")

# === One-Hot Encoding Function === #
def one_hot_label(category_id):
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    vec[category_id] = 1.0
    return vec

# === Debugging: Print One-Hot Labels for Every Category === #
print("\n[DEBUG] One-Hot Labels for Every Category:")
for cat_id in range(NUM_CLASSES):
    print(f"  Category {cat_id} → One-Hot: {one_hot_label(cat_id)}")

# === Data Generator === #
def data_generator(samples):
    for image_data, bbox, label, bbox_offset in samples:
        if isinstance(image_data, str):  # Load image from file
            image = cv2.imread(image_data)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_data  # In-memory image

        x_min, y_min, x_max, y_max = bbox
        if x_max <= x_min or y_max <= y_min:
            continue

        crop = image_rgb[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, INPUT_SIZE) / 255.0

        yield crop, one_hot_label(label), np.array(bbox_offset, dtype=np.float32)

# === Build TensorFlow Datasets === #
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_samples),
    output_signature=(
        tf.TensorSpec(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32),
        tf.TensorSpec(shape=(4,), dtype=tf.float32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_samples),
    output_signature=(
        tf.TensorSpec(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32),
        tf.TensorSpec(shape=(4,), dtype=tf.float32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === Initialize Models === #
print("\n[INFO] Initializing models...")
feature_extractor = get_feature_extractor(
    input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3),
    feature_dim=FEATURE_DIM
)

for layer in feature_extractor.layers[-FEATURE_EXTRACTOR_TRAINABLE_LAYERS:]:
    layer.trainable = True

classification_model = compile_classification_model(
    build_classification_model(FEATURE_DIM, NUM_CLASSES)
)

boundary_model = compile_boundary_correction_model(
    build_boundary_correction_model(FEATURE_DIM)
)

# === Learning Rate Scheduler === #
lr_schedule = ReduceLROnPlateau(
    monitor='accuracy', factor=0.5, patience=1, min_lr=1e-6, verbose=1
)

# === Training Loop === #
for epoch in range(NUM_EPOCHS):
    print(f"\n=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
    epoch_loss, epoch_acc, steps = 0.0, 0.0, 0
    val_loss, val_acc, val_steps = 0.0, 0.0, 0

    # Training Loop
    for batch_imgs, batch_labels, batch_bboxes in tqdm(train_dataset, desc=f"Epoch {epoch + 1}"):
        feats = feature_extractor(batch_imgs, training=True)

        c_loss, c_acc = classification_model.train_on_batch(feats, batch_labels)
        b_loss = boundary_model.train_on_batch(feats, batch_bboxes)

        epoch_loss += c_loss
        epoch_acc += c_acc
        steps += 1

    # Validation Loop
    for batch_imgs, batch_labels, batch_bboxes in val_dataset:
        feats = feature_extractor(batch_imgs, training=False)
        c_loss, c_acc = classification_model.evaluate(feats, batch_labels, verbose=0)

        val_loss += c_loss
        val_acc += c_acc
        val_steps += 1

    avg_loss = epoch_loss / steps if steps > 0 else 0
    avg_acc = epoch_acc / steps if steps > 0 else 0
    avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
    avg_val_acc = val_acc / val_steps if val_steps > 0 else 0

    print(f"[TRAIN] Classification => loss={avg_loss:.4f}, acc={avg_acc:.4f}")
    print(f"[VALID] Classification => loss={avg_val_loss:.4f}, acc={avg_val_acc:.4f}")

# === Save Models === #
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
classification_model.save(os.path.join(MODEL_SAVE_PATH, "classification_model.h5"))
boundary_model.save(os.path.join(MODEL_SAVE_PATH, "boundary_model.h5"))
print("[INFO] Training complete and models saved.")
