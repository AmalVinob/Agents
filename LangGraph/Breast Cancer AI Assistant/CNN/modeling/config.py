# Paths
IMAGE_DIR = 'C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/jpeg'
CSV_DIR = 'C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv'
TRAIN_CSV = f'{CSV_DIR}/mass_train_fixed.csv'
TEST_CSV = f'{CSV_DIR}/mass_test_fixed.csv'

# Image
IMAGE_SIZE = (224, 224, 3)

# Training
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4

# config.py
MODEL_SAVE_PATH = "../models/mobilenetv2_ddsm.h5"
CHECKPOINT_PATH = '../models/{epoch:02d}-{val_accuracy:.2f}.weights.h5'


