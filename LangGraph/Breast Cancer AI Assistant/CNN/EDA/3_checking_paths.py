import pandas as pd
import json

df_meta= pd.read_csv("C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/meta.csv")
df_dicom = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/dicom_info.csv')

cropped_images = df_dicom[df_dicom.SeriesDescription=='cropped images'].image_path
print(cropped_images.head(5))

full_mammo = df_dicom[df_dicom.SeriesDescription=='full mammogram images'].image_path
print(full_mammo.head())

roi_img = df_dicom[df_dicom.SeriesDescription=='ROI mask images'].image_path
print(roi_img.head())

imdir = 'C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/jpeg'

# change directory path of images
cropped_images = cropped_images.replace('CBIS-DDSM/jpeg', imdir, regex=True)
full_mammo = full_mammo.replace('CBIS-DDSM/jpeg', imdir, regex=True)
roi_img = roi_img.replace('CBIS-DDSM/jpeg', imdir, regex=True)

# view new paths
print('Cropped Images paths:\n')
print(cropped_images.iloc[0])
print('Full mammo Images paths:\n')
print(full_mammo.iloc[0])
print('ROI Mask Images paths:\n')
print(roi_img.iloc[0])

full_mammo_dict = dict()
cropped_images_dict = dict()
roi_img_dict = dict()



import os

# Create dictionaries for fast lookup
full_mammo_dict = dict()
cropped_images_dict = dict()
roi_img_dict = dict()

# Loop through each image path in the list and extract UID folder name
for dicom in full_mammo:
    key = os.path.basename(os.path.dirname(dicom))  # UID folder name
    full_mammo_dict[key] = dicom

for dicom in cropped_images:
    key = os.path.basename(os.path.dirname(dicom))
    cropped_images_dict[key] = dicom

for dicom in roi_img:
    key = os.path.basename(os.path.dirname(dicom))
    roi_img_dict[key] = dicom

# print(f"Full Mammo Dict keys: {list(full_mammo_dict.keys())[:5]}")

import json

# Optional: Save dictionaries to disk
with open("full_mammo_dict.json", "w") as f:
    json.dump(full_mammo_dict, f)

with open("cropped_images_dict.json", "w") as f:
    json.dump(cropped_images_dict, f)

with open("roi_img_dict.json", "w") as f:
    json.dump(roi_img_dict, f)


mass_train = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_case_description_train_set.csv')
mass_test = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv//mass_case_description_test_set.csv')

# def fix_image_path(data):
#     """Correct dicom paths to correct local image file paths"""
#     for index, img in enumerate(data.values):
#         try:
#             full_mammo_uid = os.path.basename(os.path.dirname(img[11]))
#             cropped_uid = os.path.basename(os.path.dirname(img[12]))

#             if full_mammo_uid in full_mammo_dict:
#                 data.iloc[index, 11] = full_mammo_dict[full_mammo_uid]
#             else:
#                 print(f"Missing full mammo UID: {full_mammo_uid}")

#             if cropped_uid in cropped_images_dict:
#                 data.iloc[index, 12] = cropped_images_dict[cropped_uid]
#             else:
#                 print(f"Missing cropped UID: {cropped_uid}")

#         except Exception as e:
#             print(f"Error fixing path at index {index}: {e}")

# def fix_image_path(data):
#     """Fix image paths in the DataFrame using UID-based mapping"""
#     for index, row in data.iterrows():
#         try:
#             # Extract UID string from the path (assumes it contains a UID in it)
#             full_mammo_path = row['full mammogram image file path']
#             cropped_path = row['cropped image file path']

#             full_mammo_uid = full_mammo_path.split('/')[1] if '/' in full_mammo_path else None
#             cropped_uid = cropped_path.split('/')[1] if '/' in cropped_path else None

#             if full_mammo_uid in full_mammo_dict:
#                 data.at[index, 'full mammogram image file path'] = full_mammo_dict[full_mammo_uid]
#             else:
#                 print(f"Missing full mammo UID: {full_mammo_uid}")

#             if cropped_uid in cropped_images_dict:
#                 data.at[index, 'cropped image file path'] = cropped_images_dict[cropped_uid]
#             else:
#                 print(f"Missing cropped UID: {cropped_uid}")

#         except Exception as e:
#             print(f"Error fixing path at index {index}: {e}")


def fix_image_path(data):
    """correct dicom paths to correct image paths"""
    for index, img in enumerate(data.values):
        img_name = img[11].split("/")[2]
        data.iloc[index,11] = full_mammo_dict[img_name]
        img_name = img[12].split("/")[2]
        data.iloc[index,12] = cropped_images_dict[img_name]
        
# apply to datasets
fix_image_path(mass_train)
fix_image_path(mass_test)

print(mass_train["cropped image file path"][0:5])

mass_train.to_csv("C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_train_fixed.csv", index=False)
mass_test.to_csv("C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_test_fixed.csv", index=False)
