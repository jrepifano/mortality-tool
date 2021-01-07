import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

id = 'subject_id'   # I get tired of typing this
labels = pd.read_csv(os.getcwd()+'/data/admissions.csv')
labels = labels[[id, 'hospital_expire_flag']]
lab_items = pd.read_csv(os.getcwd()+'/data/d_labitems.csv')
lab_items = lab_items[['itemid', 'label']]
lab_values = pd.read_csv(os.getcwd()+'/data/labevents.csv')
pass
