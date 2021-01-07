import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

def load_data():
    dir = os.getcwd()
    X_eicu = np.load(dir+'/data/X_eicu.npy')
    y_eicu = np.load(dir+'/data/y_eicu.npy')
    X_mimc = np.load(dir+'/data/X_mimic.npy')
    y_mimic = np.load(dir+'/data/y_mimic.npy')
    X = np.vstack((X_eicu, X_mimc))
    y = np.hstack((y_eicu, y_mimic))
    # imputer = IterativeImputer()
    scaler = StandardScaler()
    # X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)
    X, y = RandomUnderSampler().fit_resample(X, y)
    print(X.shape)
    imputer = KNNImputer(n_neighbors=3)
    X = imputer.fit_transform(X)
    return X, y

def load_new_data(feat_names):
    dir = os.getcwd()
    X = np.load(dir+'/data/x_eicu_all.npy')
    y = np.load(dir+'/data/y_eicu_all.npy')
    loc = np.load(dir+'/data/75_tol_feats.npy')
    X = X[:, loc]
    X = IterativeImputer(verbose=1).fit_transform(X)
    X = StandardScaler().fit_transform(X)
    return X, y[:, 1]

def main():
    # feature_names = ['Lactate', 'GCS-Day 1 Motor', 'Albumin', 'Age',
    #                  'Creatinine', 'Glucose', 'Platelet', 'PT', 'WBC', 'BUN']
    feat_names = ['lactate', 'eyes', 'oobintubday1', 'oobventday1', 'motor', 'verbal',
       'AST (SGOT)', 'phosphate', 'Base Excess', 'FiO2', 'bicarbonate',
       'PT - INR', 'intubated', 'PT', 'HCO3', 'temperature', 'creatinine_x',
       'ALT (SGPT)', 'pH', 'anion gap', 'WBC x 1000', 'meanbp', 'O2 Sat (%)',
       'albumin_y', 'BUN', 'total protein', 'gender', 'diabetes', 'albumin_x',
       'wbc', 'calcium', 'bun', 'heartrate', '-eos', 'creatinine',
       'creatinine_y', 'total bilirubin', 'PTT', '-monos', 'bilirubin',
       'potassium', 'glucose_x', '-lymphs', 'glucose_y', 'RDW', 'magnesium',
       'platelets x 1000', '-polys', 'sodium_x', 'paCO2', 'respiratoryrate',
       'paO2', 'urinary specific gravity', 'sodium_y', 'chloride',
       'metastaticcancer', 'alkaline phos.', 'troponin - I', 'hepaticfailure',
       'bedside glucose', 'cirrhosis', 'Hgb', 'immunosuppression', 'MCHC',
       'CPK', 'Hct', 'RBC', 'hematocrit', 'age', 'MPV', '-basos', 'midur',
       'leukemia', 'lymphoma', 'MCV', 'aids', 'MCH']
    X, y = load_new_data(feat_names)
    neg_idx = np.argwhere(y == 0)
    pos_idx = np.argwhere(y == 1)
    for i in range(len(feat_names)):
        neg_values = pd.DataFrame(X[neg_idx, i], columns=['Negative Class'])
        pos_values = pd.DataFrame(X[pos_idx, i], columns=['Positive Class'])
        p1 = sns.kdeplot(neg_values['Negative Class'], shade=True, color='r', legend=True)
        p2 = sns.kdeplot(pos_values['Positive Class'], shade=True, color='b', legend=True)
        plt.legend(['Negative Class', 'Positive Class'], loc='upper right')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.title(feat_names[i])
        plt.savefig(os.getcwd()+'/results/new_feat_kdes/'+feat_names[i])
        plt.clf()
        plt.cla()
        plt.close()


if __name__ == '__main__':
    main()
