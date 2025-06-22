import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns


df_meta= pd.read_csv("C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/meta.csv")
df_dicom = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/dicom_info.csv')


mass_train_fixed = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_train_fixed.csv')
mass_test_fixed = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_test_fixed.csv')

print(mass_train_fixed.pathology.unique())

mass_train_fixed = mass_train_fixed.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

mass_train_fixed.head(5)

mass_train_fixed['mass_shape'] = mass_train_fixed['mass_shape'].bfill()
mass_train_fixed['mass_margins'] = mass_train_fixed['mass_margins'].bfill()

mass_test_fixed = mass_test_fixed.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})
mass_test_fixed['mass_margins'] = mass_test_fixed['mass_margins'].bfill()


value = mass_train_fixed['pathology'].value_counts()
plt.figure(figsize=(8,6))

plt.pie(value, labels=value.index, autopct='%1.1f%%')
plt.title('Breast Cancer Mass Types', fontsize=14)
plt.savefig('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/CNN/EDA/pathology_distributions_red.png')
plt.show()

plt.figure(figsize=(8,6))

sns.countplot(mass_train_fixed, x='breast_density', hue='pathology')
plt.title('Breast Density vs Pathology\n\n1: fatty || 2: Scattered Fibroglandular Density\n3: Heterogenously Dense || 4: Extremely Dense',
          fontsize=14)
plt.xlabel('Density Grades')
plt.ylabel('Count')
plt.legend()
plt.savefig('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/CNN/EDA/density_pathology_red.png')
plt.show()


mass_train_fixed.to_csv("C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_train_fixed.csv", index=False)
mass_test_fixed.to_csv("C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/mass_test_fixed.csv", index=False)
