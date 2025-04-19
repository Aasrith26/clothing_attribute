import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models

# -----------------------------
# Configuration
# -----------------------------
PRETRAINED_MODEL_PATH = "VGG16_clothing_attribute_model_continued.keras"  # Path of your previously saved model
DATASET_PATH = "dataset2/crops"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
ADDITIONAL_EPOCHS = 10
VAL_SPLIT = 0.1
FINE_TUNE_LR = 1e-4               # Learning rate for further fine-tuning
UNFREEZE_LAYERS = 8               # Number of layers to unfreeze (or keep the same)
MODEL_SAVE_PATH = "VGG16_clothing_attribute_model_final_1.keras"

print("[DEBUG] Using Additional Epochs:", ADDITIONAL_EPOCHS)
print("[DEBUG] Dataset Path:", DATASET_PATH)


train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest",
    validation_split=VAL_SPLIT
)

train_ds = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_ds = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

num_classes = len(train_ds.class_indices)
print(f"[DEBUG] Number of Classes Detected: {num_classes}")

# -----------------------------
# Load Previously Trained Model
# -----------------------------
print(f"[DEBUG] Loading previously trained model from {PRETRAINED_MODEL_PATH}...")
model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
print("[DEBUG] Model loaded successfully.")


# Identify the base model layers (assuming a VGG16 architecture)
# Typically, the first part of the model is VGG16, then your custom head.
# We'll unfreeze the last 'UNFREEZE_LAYERS' layers in the base VGG16 portion.

# Let's find which layers belong to the VGG16 base.
# Often you can identify them by name: e.g. 'block5_conv2', 'block5_conv3', etc.
base_model_layers = []
for i, layer in enumerate(model.layers):
    if "block" in layer.name or "vgg" in layer.name:
        base_model_layers.append(layer)

# Now unfreeze the last UNFREEZE_LAYERS in 'base_model_layers'
print(f"[DEBUG] Unfreezing the last {UNFREEZE_LAYERS} layers in the base VGG16 portion.")
for layer in base_model_layers[-UNFREEZE_LAYERS:]:
    layer.trainable = True

# Re-compile with a lower LR for continued fine-tuning
model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("[DEBUG] Model summary AFTER unfreezing layers:")
model.summary()

# -----------------------------
# Continue Training (No EarlyStopping)
# -----------------------------
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-6
)

print(f"[DEBUG] Resuming training for {ADDITIONAL_EPOCHS} more epochs...")
history = model.fit(
    train_ds,
    epochs=ADDITIONAL_EPOCHS,
    validation_data=val_ds,
    callbacks=[reduce_lr]  # We keep only ReduceLROnPlateau
)

# -----------------------------
# Save Updated Model
# -----------------------------
print(f"[DEBUG] Saving new model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("[INFO] Continued training complete. Model saved.")
