# Load libraries (you already have this part covered nicely)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Load fixed CSVs
mass_train_fixed = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_train_fixed.csv')
mass_test_fixed = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_test_fixed.csv')


# Image preprocessor
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

# Combine and preprocess data
full_mass = pd.concat([mass_train_fixed, mass_test_fixed], axis=0)
target_size = (224, 224, 3)
sample_size = 1696
full_mass_sample = full_mass.sample(n=sample_size, random_state=42)
full_mass_sample['processed_images'] = full_mass_sample['image_file_path'].apply(lambda x: image_processor(x, target_size))

class_mapper = {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}
full_mass_sample['labels'] = full_mass_sample['pathology'].replace(class_mapper)

X = np.array(full_mass_sample['processed_images'].tolist())
y = full_mass_sample['labels'].values

# Just do 1 fold now to test loading + evaluating
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_index, test_index = next(iter(kf.split(X)))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# Optionally save test set for reuse
np.save('./X_test.npy', X_test)
np.save('./y_test.npy', y_test)


# Build the model
base_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# 🔥 Load best model weights
model.load_weights('./model_2/model-46-0.70.weights.h5')

# ✅ Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print("✅ Final model accuracy on test set:", acc)
