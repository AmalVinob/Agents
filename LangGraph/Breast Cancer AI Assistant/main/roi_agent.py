import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Dropout
from keras.regularizers import l1_l2

# === Image Preprocessing ===
# def image_processor(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError(f"Image not found: {image_path}")
#     image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
#     image = cv2.resize(image, (224, 224))
#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#     image = image.astype("float32") / 255.0
#     return np.expand_dims(image, axis=0)
def image_processor(image_path, target_size):
    absolute_image_path = os.path.abspath(image_path)
    image = cv2.imread(absolute_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size[1], target_size[0]))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    image = cv2.merge((l_channel, a_channel, b_channel))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    image = 255 - image  # negative
    image = image.astype(np.float32) / 255.0
    return image

# === Build Model ===
def build_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# === Classify ROI ===
def classify_roi_image(image_path):
    model = build_model()
    weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/model-46-0.70.weights.h5"))
    model.load_weights(weights_path)
    image = image_processor(image_path, (224, 224, 3))  # ✅ fixed
    image = np.expand_dims(image, axis=0)  # ✅ model expects a batch
    prediction = model.predict(image)[0][0]
    label = "MALIGNANT" if prediction >= 0.5 else "BENIGN"
    return {
        "image": os.path.basename(image_path),
        "prediction": label,
        "confidence": round(float(prediction), 4)
    }
