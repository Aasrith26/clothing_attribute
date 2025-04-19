import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import AdamW

# Define Constants
IMG_SIZE = (299, 299)  # InceptionResNetV2 standard size
BATCH_SIZE = 32
NUM_EPOCHS = 30  # 30 epochs as you requested
CROPS_DIR = "dataset2/crops"  # Update with your dataset path

# Debugging Statements
print("[DEBUG] Using Image Size:", IMG_SIZE)
print("[DEBUG] Batch Size:", BATCH_SIZE)
print("[DEBUG] Dataset Path:", CROPS_DIR)

# Data Augmentation (with a 90:10 train:validation split)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest",
    validation_split=0.1  # 90% training, 10% validation
)

train_ds = train_datagen.flow_from_directory(
    CROPS_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_ds = train_datagen.flow_from_directory(
    CROPS_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Auto-detect number of classes
NUM_CLASSES = len(train_ds.class_indices)
print(f"[DEBUG] Number of Detected Classes: {NUM_CLASSES}")

# --------------------------------------------
# Custom L‑Softmax Loss (Simplified Implementation)
# --------------------------------------------
import tensorflow.keras.backend as K


def lsoftmax_loss(margin=4):
    """
    A simplified version of the L-Softmax loss.
    NOTE: This implementation assumes that the final dense layer
    produces logits that are in the range [-1,1]. In practice, the L-Softmax
    loss should be applied to normalized features and weights.
    """

    def loss_fn(y_true, y_pred):
        # Clip logits for safety in arccos computation.
        y_pred_clipped = tf.clip_by_value(y_pred, -1 + 1e-7, 1 - 1e-7)

        # Get the index of the true class for each sample.
        y_true_index = tf.argmax(y_true, axis=1)
        # Gather logits for the true class.
        logits_true = tf.gather(y_pred_clipped, y_true_index, axis=1, batch_dims=1)

        # Compute theta for the true class: theta = arccos(logit)
        theta_true = tf.acos(logits_true)

        # Compute psi(theta) using the Chebyshev polynomial for margin m = 4:
        # psi(theta) = 8*cos^4(theta) - 8*cos^2(theta) + 1
        psi_theta = 8 * tf.pow(tf.cos(theta_true), 4) - 8 * tf.pow(tf.cos(theta_true), 2) + 1

        # Replace the true class logits with psi(theta) in the logits tensor.
        # Create indices for scatter update.
        indices = tf.stack([tf.range(tf.shape(y_pred)[0]), y_true_index], axis=1)
        # Modify logits: this creates a new tensor with updated true-class values.
        y_pred_modified = tf.tensor_scatter_nd_update(y_pred_clipped, indices, psi_theta)

        # Compute softmax cross entropy with the modified logits.
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred_modified)
        return loss

    return loss_fn


# --------------------------------------------
# Build InceptionResNetV2 Model with L-Softmax Loss
# --------------------------------------------

# Load base model with pretrained ImageNet weights.
base_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False  # Freeze base model initially

# Build the custom classification head.
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
# IMPORTANT: Remove the softmax activation so that we output logits.
outputs = layers.Dense(NUM_CLASSES, activation="linear")(x)

# Create the model.
model = models.Model(inputs=base_model.input, outputs=outputs)

# Compile the model with AdamW optimizer and the custom L-Softmax loss.
model.compile(
    optimizer=AdamW(learning_rate=0.001),
    loss=lsoftmax_loss(margin=4),
    metrics=["accuracy"]
)

# Fine-tuning: Unfreeze the last 50 layers of the base model.
print("[DEBUG] Unfreezing last 50 layers for fine-tuning...")
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Re-compile with a lower learning rate for fine-tuning.
model.compile(
    optimizer=AdamW(learning_rate=0.0001),
    loss=lsoftmax_loss(margin=4),
    metrics=["accuracy"]
)

# Set up callbacks.
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1)

print("[DEBUG] Training Model with L-Softmax loss for 30 epochs...")
history = model.fit(
    train_ds,
    epochs=NUM_EPOCHS,
    validation_data=val_ds,
    callbacks=[lr_scheduler, early_stopping]
)

# Save the model.
model_save_path = "InceptionResNetV2_lsoftmax_clothing_attribute_model.keras"
print(f"[DEBUG] Training Completed. Saving Model as {model_save_path}")
model.save(model_save_path)
print("[DEBUG] Model saved successfully!")
