import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

# CONFIGURATION
MODEL_LOAD_PATH = "clothing_attribute_model.keras"
IMG_SIZE = (299, 299)

# Category mapping (model outputs indices 0-6, so we add 1 to match these keys)
category_dict = {
    1: 'Dress',
    2: 'Outerwear',
    3: 'Shirt',
    4: 'Suit',
    5: 'Sweater',
    6: 'Tank Top',
    7: 'T-shirt'
}


def predict_image(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image:", image_path)
        return None, []

    # Convert BGR to RGB and resize the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)

    # Preprocess image for InceptionResNetV2
    img_preprocessed = preprocess_input(img_resized.astype(np.float32))
    input_tensor = np.expand_dims(img_preprocessed, axis=0)

    # Load the saved model
    model = tf.keras.models.load_model(MODEL_LOAD_PATH)

    # Perform prediction
    preds = model.predict(input_tensor)
    pred_index = np.argmax(preds, axis=1)[0]
    confidence = preds[0][pred_index]

    # Adjust index: model outputs 0-6; add 1 to match our category_dict keys.
    pred_class = pred_index + 1
    predicted_category = category_dict.get(pred_class, "Unknown")

    return img, [(predicted_category, confidence)]


def main():
    # Specify a test image path (update this path as needed)
    test_img_path = "dataset2/crops/3/481_3_45_114.jpg"
    img, predictions = predict_image(test_img_path)
    if img is None:
        return

    print("Predictions (category, confidence):", predictions)

    # Display the image with predicted class name and confidence
    display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pred_cat, conf = predictions[0]
    cv2.putText(display_img, f"Class: {pred_cat} ({conf:.2f})",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("Test Image", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
