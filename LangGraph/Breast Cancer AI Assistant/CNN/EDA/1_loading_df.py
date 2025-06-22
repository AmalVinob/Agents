import pandas as pd
import numpy as np

df_meta= pd.read_csv("C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/meta.csv")
df_meta.head()

df_meta.info()

df_dicom = pd.read_csv('C:/Users/1090135/Downloads/orion_learning/agents/autogen/Breast_cancer/datasets/image_data/csv/dicom_info.csv')
df_dicom.head()
df_dicom.info()

print(df_dicom.SeriesDescription.unique())