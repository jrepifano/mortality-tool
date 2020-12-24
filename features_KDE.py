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


def main():
    feature_names = ['Lactate', 'GCS-Day 1 Motor', 'Albumin', 'Age',
                     'Creatinine', 'Glucose', 'Platelet', 'PT', 'WBC', 'BUN']
    X, y = load_data()
    neg_idx = np.argwhere(y == 0)
    pos_idx = np.argwhere(y == 1)
    for i in range(len(feature_names)):
        neg_values = pd.DataFrame(X[neg_idx, i], columns=['Negative Class'])
        pos_values = pd.DataFrame(X[pos_idx, i], columns=['Positive Class'])
        p1 = sns.kdeplot(neg_values['Negative Class'], shade=True, color='r', legend=True)
        p2 = sns.kdeplot(pos_values['Positive Class'], shade=True, color='b', legend=True)
        plt.legend(['Negative Class', 'Positive Class'], loc='upper left')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.title(feature_names[i])
        plt.savefig(os.getcwd()+'/results/feature_kdes/'+feature_names[i]+'undersample-knn')
        plt.clf()
        plt.cla()
        plt.close()


if __name__ == '__main__':
    main()
