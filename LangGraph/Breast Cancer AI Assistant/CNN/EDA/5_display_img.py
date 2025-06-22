import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg


with open('full_mammo_dict.json') as f:
    full_mammo_dict = json.load(f)

mass_train_fixed = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_train_fixed.csv')
mass_test_fixed = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_test_fixed.csv')


def display_images(column, number):
    # create figure and axes
    number_to_visualize = number
    rows = 1
    cols = number_to_visualize
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    
    # Loop through rows and display images
    for index, row in mass_train_fixed.head(number_to_visualize).iterrows():
        image_path = row[column]
        image = mpimg.imread(image_path)
        ax = axes[index]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"{row['pathology']}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

print('Full Mammograms:\n')
display_images('image_file_path', 5)
print('Cropped Mammograms:\n')
display_images('cropped_image_file_path', 5)