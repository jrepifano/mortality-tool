import os
import torch
import scipy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Model(torch.nn.Module):
    def __init__(self, layer_1, layer_2, layer_3):
        super(Model, self).__init__()
        self.lin1 = torch.nn.Linear(12, layer_1)
        self.lin2 = torch.nn.Linear(layer_1, layer_2)
        self.lin3 = torch.nn.Linear(layer_2, layer_3)
        self.lin4 = torch.nn.Linear(layer_3, 1)
        self.selu = torch.nn.SELU()

    def forward(self, x):
        x = self.selu(self.lin1(x))
        x = self.selu(self.lin2(x))
        x = self.selu(self.lin3(x))
        x = self.lin4(x)
        return x


def load_data():
    cd = os.getcwd()
    x_eicu = pd.read_csv('data/x_eicu.csv')
    y_eicu = pd.read_csv('data/y_eicu.csv')
    mimic = pd.read_csv('data/mimic.csv')
    assert np.all(x_eicu['patientunitstayid'].to_numpy() == y_eicu['patientunitstayid'].to_numpy())
    feature_list = ['lactate', 'oobventday1', 'eyes', 'motor', 'verbal', 'albumin_x',
                    'age', 'creatinine_x', 'BUN', 'PT - INR', 'WBC x 1000', 'meanbp']
    feature_list_mimic = ['Lactate', 'firstdayvent', 'gcseyes', 'gcsmotor', 'gcsverbal', 'Albumin',
                          'Age', 'Creatinine', 'BUN', 'INR', 'WBC', 'MAP']
    x_eicu = x_eicu[feature_list].to_numpy()
    y_eicu = y_eicu['actualicumortality'].to_numpy()
    x_mimic = mimic[feature_list_mimic].to_numpy()
    y_mimic = mimic['Mortality'].to_numpy()
    x = np.vstack((x_eicu, x_mimic))
    y = np.hstack((y_eicu, y_mimic))
    shuffler = np.random.permutation(len(x))
    return x[shuffler], y[shuffler]


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def main():
    x, y = load_data()
    kfold = StratifiedKFold(n_splits=10)
    logits_all = []
    labels_all = []
    accuracy = []
    precision = []
    sensitivity = []
    specificity = []
    roc_auc = []
    prc_auc = []
    balanced_acc = []
    pos_likelihood_ratio = []
    neg_likelihood_ratio = []
    counter = 1
    for train_index, test_index in kfold.split(x, y):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        imputer = IterativeImputer()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(x_train))
        x_test = scaler.transform(imputer.transform(x_test))
        x_train, y_train = torch.from_numpy(x_train).float().to('cuda:0'), torch.from_numpy(y_train).float().to('cuda:0')
        model = Model(197, 198, 112)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([14.80], device='cuda:0'))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03104, weight_decay=0.01043, momentum=0.4204, nesterov=True)
        model.train()
        model.to('cuda:0')
        no_epochs = 127
        for epoch in range(no_epochs):
            optimizer.zero_grad()
            outputs = model.forward(x_train)
            loss = criterion(outputs, y_train.view(-1, 1))
            loss.backward()
            optimizer.step()
        model.eval()
        outputs = model.forward(torch.from_numpy(x_test).float().to('cuda:0'))
        logits = torch.sigmoid(outputs).detach().cpu().numpy()
        logits_all.append(logits.reshape(-1))
        labels_all.append(y_test)
        print('Iter {}/10 done'.format(counter))
        counter += 1
    # logits_all = np.hstack(logits_all)
    # labels_all = np.hstack(labels_all)
    for i in range(len(logits_all)):
        tn, fp, fn, tp = confusion_matrix(labels_all[i], np.round(logits_all[i])).ravel()
        accuracy.append((tp + tn) / (tp + tn + fp + fn))
        precision.append(tp / (tp + fp))
        sensitivity.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))
        roc_auc.append(roc_auc_score(labels_all[i], logits_all[i]))
        prc_auc.append(average_precision_score(labels_all[i], logits_all[i]))
        balanced_acc.append(balanced_accuracy_score(labels_all[i], np.round(logits_all[i])))
    mean, confidence_interval = mean_confidence_interval(accuracy)
    print('Accuracy Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))
    mean, confidence_interval = mean_confidence_interval(precision)
    print('Precision Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))
    mean, confidence_interval = mean_confidence_interval(sensitivity)
    print('Sensitivity Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))
    mean, confidence_interval = mean_confidence_interval(specificity)
    print('Specificity Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))
    mean, confidence_interval = mean_confidence_interval(roc_auc)
    print('ROC_AUC Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))
    mean, confidence_interval = mean_confidence_interval(prc_auc)
    print('PRC_AUC Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))
    mean, confidence_interval = mean_confidence_interval(balanced_acc)
    print('Balanced Accuracy Mean and confidence interval: {:4f}, {:4f}'.format(mean, confidence_interval))


if __name__ == '__main__':
    main()
