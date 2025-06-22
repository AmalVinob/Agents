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

for dicom in full_mammo:
    key = dicom.split("/")[4]
    full_mammo_dict[key] = dicom
for dicom in cropped_images:
    key = dicom.split("/")[4]
    cropped_images_dict[key] = dicom
for dicom in roi_img:
    key = dicom.split("/")[4]
    roi_img_dict[key] = dicom


next(iter((full_mammo_dict.items())))



with open('full_mammo_dict.json', 'w') as f:
    json.dump(full_mammo_dict, f)

with open('cropped_images_dict.json', 'w') as f:
    json.dump(cropped_images_dict, f)

with open('roi_img_dict.json', 'w') as f:
    json.dump(roi_img_dict, f)


# with open('full_mammo_dict.json') as f:
#     full_mammo_dict = json.load(f)


mass_train = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_case_description_train_set.csv')
mass_test = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv//mass_case_description_test_set.csv')

print(mass_train.head())

# def fix_image_path(data):
#     """correct dicom paths to correct image paths"""
#     for index, img in enumerate(data.values):
#         img_name = img[11].split("/")[2]
#         data.iloc[index,11] = full_mammo_dict[img_name]
#         img_name = img[12].split("/")[2]
#         data.iloc[index,12] = cropped_images_dict[img_name]
        
# # apply to datasets
# fix_image_path(mass_train)
# fix_image_path(mass_test)

print(mass_train["cropped image file path"][0])

# def fix_image_path(data):
#     """correct dicom paths to actual image file paths"""
#     for index, img in enumerate(data.values):
#         full_mammo_uid = img[11].split("/")[2]
#         cropped_uid = img[12].split("/")[2]

#         if full_mammo_uid in full_mammo_dict:
#             data.iloc[index, 11] = full_mammo_dict[full_mammo_uid]
#         else:
#             print(f"Missing full mammo UID: {full_mammo_uid}")

#         if cropped_uid in cropped_images_dict:
#             data.iloc[index, 12] = cropped_images_dict[cropped_uid]
#         else:
#             print(f"Missing cropped image UID: {cropped_uid}")

fix_image_path(mass_train)
fix_image_path(mass_test)