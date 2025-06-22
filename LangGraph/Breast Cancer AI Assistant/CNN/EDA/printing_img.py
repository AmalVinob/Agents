import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

mass_train = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_case_description_train_set.csv')
print(os.path.exists(mass_train["cropped image file path"][0])) 



img_path = mass_train["cropped image file path"][0]
img = mpimg.imread(img_path)
plt.imshow(img, cmap='gray')
plt.title("Cropped Mammogram")
plt.axis('off')
plt.show()
