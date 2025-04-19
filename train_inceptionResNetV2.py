import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

# -----------------------------
# Configuration
# -----------------------------
PRETRAINED_MODEL_PATH = "InceptionResNetV2_clothing_attribute_model.keras"  # Previous model
IMG_SIZE = (299, 299)       # InceptionResNetV2 standard size
BATCH_SIZE = 32
ADDITIONAL_EPOCHS = 10      # Number of extra epochs
DATASET_PATH = "dataset2/crops"
VAL_SPLIT = 0.2
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "InceptionResNetV2_clothing_attribute_model_4.keras"  # New model path

# -----------------------------
# Data Augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1.0/255,
    validation_split=VAL_SPLIT
)

train_ds = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_ds = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True
)

# -----------------------------
# STEP 1: Load Previously Trained Model
# -----------------------------
print(f"[DEBUG] Loading previously trained model from {PRETRAINED_MODEL_PATH}...")
model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
print("[DEBUG] Model loaded successfully.")

model.summary()  # Optional: verify layers

# -----------------------------
# STEP 2: Compile Model Again
# -----------------------------
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# STEP 3: Continue Training
# -----------------------------
print(f"[DEBUG] Resuming training for an additional {ADDITIONAL_EPOCHS} epochs...")
history = model.fit(
    train_ds,
    epochs=ADDITIONAL_EPOCHS,
    validation_data=val_ds
)

# -----------------------------
# STEP 4: Save Updated Model
# -----------------------------
model.save(MODEL_SAVE_PATH)
print(f"[INFO] Updated model saved to: {MODEL_SAVE_PATH}")
