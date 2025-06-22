import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import image_processor
from config import IMAGE_SIZE, TRAIN_CSV, TEST_CSV

class_mapper = {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}

def load_dataset():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    full_df = pd.concat([train_df, test_df])

    full_df['processed'] = full_df['image_file_path'].apply(lambda x: image_processor(x, IMAGE_SIZE))
    full_df['labels'] = full_df['pathology'].replace(class_mapper)

    X = np.array(full_df['processed'].tolist())
    y = full_df['labels'].values

    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_augmentation():
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )