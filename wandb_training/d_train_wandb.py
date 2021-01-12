import os
import torch
import wandb
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--layer_1', help='layer 1 size', type=int, default=50, required=True)
    parser.add_argument(
        '--layer_2', help='layer 2 size', type=int, default=50, required=True)
    parser.add_argument(
        '--layer_3', help='layer 3 size', type=int, default=50, required=True)
    parser.add_argument(
        '--lr', help='Learning Rate', type=float, default=0.01, required=True)
    parser.add_argument(
        '--weight_decay', help='Weight Decay', type=float, default=0.9, required=True)
    parser.add_argument(
        '--momentum', help='Weight Decay', type=float, default=0.9, required=True)
    parser.add_argument(
        '--no_epoch', help='Number of Epochs', type=int, default=1000, required=True)
    args = parser.parse_args()
    return args


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
    x_eicu = pd.read_csv(cd+'../data/x_eicu.csv')
    y_eicu = pd.read_csv(cd+'../data/y_eicu.csv')
    mimic = pd.read_csv(cd+'../data/mimic.csv')
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
    return x, y


def main():
    wandb.init(project='mortality-tool-newfeats')
    args = parse_args()
    x, y = load_data()
    kfold = StratifiedKFold(n_splits=10)
    logits_all = []
    labels_all = []
    counter = 1
    for train_index, test_index in kfold.split(x, y):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        imputer = IterativeImputer()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(x_train))
        x_test = scaler.transform(imputer.transform(x_test))
        x_train, y_train = torch.from_numpy(x_train).float().to('cuda:0'), torch.from_numpy(y_train).float().to('cuda:0')
        model = Model(args.layer_1, args.layer_2, args.layer_3)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([14.80], device='cuda:0'))
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
        model.train()
        model.to('cuda:0')
        for epoch in range(args.no_epoch):
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
    logits_all = np.hstack(logits_all)
    labels_all = np.hstack(labels_all)
    tn, fp, fn, tp = confusion_matrix(labels_all, np.round(logits_all)).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    roc_auc = roc_auc_score(labels_all, logits_all)
    prc_auc = average_precision_score(labels_all, logits_all)
    balanced_acc = balanced_accuracy_score(labels_all, np.round(logits_all))
    pos_likelihood_ratio = sensitivity / (1 - specificity)
    neg_likelihood_ratio = (1 - sensitivity) / specificity
    class_names = ['ALIVE', 'EXPIRED']
    wandb.log({'accuracy': accuracy, 'precision': precision, 'sensitivity': sensitivity, 'specificitiy': specificity,
               'roc_auc': roc_auc, 'prc_auc': prc_auc, 'balanced_accuracy': balanced_acc,
               'neg_likelihood_ratio': neg_likelihood_ratio, 'pos_likelihood_ratio': pos_likelihood_ratio})


if __name__ == '__main__':
    main()
