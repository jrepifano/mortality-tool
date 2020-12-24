import os
import numpy as np
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score


def load_data(split=False, resample=False):
    dir = os.getcwd()
    X_eicu = np.load(dir+'/data/X_eicu.npy')
    y_eicu = np.load(dir+'/data/y_eicu.npy')
    X_mimc = np.load(dir+'/data/X_mimic.npy')
    y_mimic = np.load(dir+'/data/y_mimic.npy')
    X = np.vstack((X_eicu, X_mimc))
    y = np.hstack((y_eicu, y_mimic))
    imputer = IterativeImputer()
    scaler = StandardScaler()
    if split == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_train = imputer.fit_transform(X_train)
        X_train = scaler.fit_transform(X_train)
        decomp = FastICA()
        X_train = decomp.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        X_test = scaler.transform(X_test)
        X_test = decomp.transform(X_test)
        if resample == True:
            X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)
            print('Shape after resampling: '+str(X_train.shape))
        return X_train, X_test, y_train, y_test
    else:
        X = imputer.fit_transform(X)
        X = scaler.fit_transform(X)
        if resample == True:
            X, y = RandomUnderSampler().fit_resample(X, y)
        return X, y

def main():
    np.random.seed(1234567890)
    split = True
    resample = True
    if split == True:
        X_train, X_test, y_train, y_test = load_data(split, resample)
    else:
        X_train, y_train = load_data(split, resample)
    log_reg = LogisticRegression(C=1000).fit(X_train, y_train)

    logits = log_reg.predict_proba(X_train)
    y_pred = log_reg.predict(X_train)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    roc_auc = roc_auc_score(y_train, logits[:, 1])
    prc_auc = average_precision_score(y_train, logits[:, 1])
    balanced_acc = balanced_accuracy_score(y_train, y_pred)
    print('---------------------LOGISTIC REGRESION TRAIN---------------------')
    print('Acc: {:.2f}, Precision: {:.2f}, Sensitivity: {:.2f}, Specificity: {:.2f}'.format(
        accuracy, precision, recall, specificity))
    print('Balanced Acc: {:.2f}, ROC AUC: {:.2f}, PRC AUC: {:.2f}'.format(balanced_acc, roc_auc, prc_auc))

    logits = log_reg.predict_proba(X_test)
    y_pred = log_reg.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    roc_auc = roc_auc_score(y_test, logits[:, 1])
    prc_auc = average_precision_score(y_test, logits[:, 1])
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print('---------------------LOGISTIC REGRESION TEST---------------------')
    print('Acc: {:.2f}, Precision: {:.2f}, Sensitivity: {:.2f}, Specificity: {:.2f}'.format(
        accuracy, precision, recall, specificity))
    print('Balanced Acc: {:.2f}, ROC AUC: {:.2f}, PRC AUC: {:.2f}'.format(balanced_acc, roc_auc, prc_auc))



if __name__ == '__main__':
    main()
