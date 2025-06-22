# Standard library
import os
import json

# Data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

# Image processing
import cv2

# TensorFlow/Keras for deep learning
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.regularizers import l1_l2

# Scikit-learn for splitting and cross-validation
from sklearn.model_selection import KFold, train_test_split


mass_train_fixed = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_train_fixed.csv')
mass_test_fixed = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_test_fixed.csv')

# Preprocessing function for input images
def image_processor(image_path, target_size):
    """Preprocess images for CNN model"""
    absolute_image_path = os.path.abspath(image_path)
    image = cv2.imread(absolute_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    image = cv2.resize(image, (target_size[1], target_size[0]))

    # Histogram Equalization (CLAHE)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    image = cv2.merge((l_channel, a_channel, b_channel))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

    # Negative transformation
    image = 255 - image

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


# Combine train and test sets
full_mass = pd.concat([mass_train_fixed, mass_test_fixed], axis=0)

# Target image size
target_size = (224, 224, 3)

# Sample a subset of data
sample_size = 1696  # Change based on available memory
full_mass_sample = full_mass.sample(n=sample_size, random_state=42)

# Apply image processing
full_mass_sample['processed_images'] = full_mass_sample['image_file_path'].apply(
    lambda x: image_processor(x, target_size)
)

# Map class labels to binary values
class_mapper = {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}
full_mass_sample['labels'] = full_mass_sample['pathology'].replace(class_mapper).infer_objects(copy=False)

# Features and labels
X = np.array(full_mass_sample['processed_images'].tolist())
y = full_mass_sample['labels'].values
num_classes = len(np.unique(y))

# K-Fold configuration
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Model hyperparameters
l1_reg = 0.001
l2_reg = 0.001
dropout_rate = 0.3
initial_learning_rate = 0.0001
batch_size = 20
max_epochs = 50
steps_per_epoch = 100

# Learning rate schedule function
def lr_schedule(epoch):
    lr = initial_learning_rate
    if epoch > 10:
        lr *= 0.1
    elif epoch > 5:
        lr *= 0.5
    return lr

# K-Fold cross-validation training loop
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\nTraining fold {fold + 1}/{n_splits}")

    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    augmented_data_generator = datagen.flow(X_train, y_train, batch_size=batch_size)

    # Base model
    base_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

    # Top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # Final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile
    model.compile(optimizer=Adam(learning_rate=initial_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    checkpoint_filepath = f'./model_fold_{fold+1}/model-{{epoch:02d}}-{{val_accuracy:.2f}}.weights.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train
    history = model.fit(
        augmented_data_generator,
        epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test, y_test),
        callbacks=[LearningRateScheduler(lr_schedule), early_stopping, model_checkpoint_callback]
    )

    # Evaluation metrics
    train_loss = np.mean(history.history['loss'])
    val_loss = np.mean(history.history['val_loss'])
    train_acc = np.mean(history.history['accuracy'])
    val_acc = np.mean(history.history['val_accuracy'])

    print(f"Fold {fold+1} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Display a few sample images from the training set
    if fold == 0:  # Only show on first fold to avoid repetitive plotting
        class_names = ['BENIGN', 'MALIGNANT']
        sample_images = X_train[:6]
        sample_labels = y_train[:6]
        titles = [class_names[label] for label in sample_labels]

        def display_images(images, titles, rows, cols, figsize=(10, 8)):
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            for i, ax in enumerate(axes.flat):
                ax.imshow(images[i])
                ax.set_title(titles[i])
                ax.axis('off')
            plt.tight_layout()
            plt.show()

        display_images(sample_images, titles, 2, 3)
