import os
import torch
import scipy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from VDP_Layers import VDP_FullyConnected, VDP_Relu, VDP_Softmax
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class VDPNet(torch.nn.Module):
    def __init__(self, layer_1, layer_2, layer_3):
        super(VDPNet, self).__init__()
        self.fullyCon1 = VDP_FullyConnected(12, layer_1, input_flag=True)
        self.fullyCon2 = VDP_FullyConnected(layer_1, layer_2, input_flag=False)
        self.fullyCon3 = VDP_FullyConnected(layer_2, layer_3, input_flag=False)
        self.fullyCon4 = VDP_FullyConnected(layer_3, 1, input_flag=False)
        self.relu = VDP_Relu()  # Actually SELU
        self.bn1 = torch.nn.BatchNorm1d(layer_1)
        self.bn2 = torch.nn.BatchNorm1d(layer_2)
        self.bn3 = torch.nn.BatchNorm1d(layer_3)
        self.bn4 = torch.nn.BatchNorm1d(1)

        self.register_buffer("thing", torch.tensor(1e-3).repeat([self.fullyCon3.out_features]))

    def forward(self, x):
        # flat_x = torch.flatten(x_input, start_dim=1)
        mu, sigma = self.fullyCon1.forward(x)
        mu = self.bn1(mu)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.fullyCon2.forward(mu, sigma)
        mu = self.bn2(mu)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.fullyCon3(mu, sigma)
        mu = self.bn3(mu)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.fullyCon4(mu, sigma)
        mu = self.bn4(mu)
        return mu, sigma

    def nll_gaussian(self, y_pred_mean, y_pred_sd, y_test):
        thing = torch.tensor(1e-3)
        # dense_label = torch.argmax(y_test, dim=1)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        y_pred_sd_inv = torch.inverse(
            y_pred_sd + torch.diag(thing.repeat([self.fullyCon4.out_features])).to(y_pred_sd.device))
        mu_ = criterion(y_pred_mean, y_test)
        mu_sigma = torch.bmm(mu_.unsqueeze(1), y_pred_sd_inv)
        ms = 0.5 * mu_sigma + 0.5 * torch.log(torch.det(y_pred_sd +
                                                        torch.diag(thing.repeat([self.fullyCon4.out_features])).to(y_pred_sd.device))).unsqueeze(1)
        ms = ms.mean()
        return ms

    def batch_loss(self, output_mean, output_sigma, label):
        output_sigma_clamp = torch.clamp(output_sigma, 1e-10, 1e+10)
        tau = 0.002
        log_likelihood = self.nll_gaussian(output_mean, output_sigma_clamp, label)
        loss_value = log_likelihood + tau * (self.fullyCon1.kl_loss_term() + self.fullyCon2.kl_loss_term() +
                                             self.fullyCon3.kl_loss_term() + self.fullyCon4.kl_loss_term())
        return loss_value


class data_loader(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().to('cuda:0')
        self.y = torch.from_numpy(y).float().to('cuda:0')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        target = self.y[index]
        data_val = self.X[index, :]
        return data_val, target


def load_data():
    cd = os.getcwd()
    x_eicu = pd.read_csv(cd+'/data/x_eicu.csv')
    y_eicu = pd.read_csv(cd+'/data/y_eicu.csv')
    mimic = pd.read_csv(cd + '/data/mimic.csv')
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
    # return x_eicu, y_eicu


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
    counter = 1
    for train_index, test_index in kfold.split(x, y):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        imputer = IterativeImputer()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(x_train))
        x_test = scaler.transform(imputer.transform(x_test))
        x_train, y_train = RandomUnderSampler().fit_resample(x_train, y_train)
        trainset = data_loader(x_train, y_train)
        testset = data_loader(x_test, y_test)
        trainloader = DataLoader(trainset, batch_size=1000, shuffle=True)
        testloader = DataLoader(testset, batch_size=1000, shuffle=True)
        model = VDPNet(31, 93, 94)
        no_epochs = 18
        optimizer = torch.optim.SGD(model.parameters(), lr=0.002219, weight_decay=0.006427, momentum=0.7589, nesterov=True)
        model.train()
        model.to('cuda:0')
        for epoch in range(no_epochs):
            for itr, (x_train, y_train) in enumerate(trainloader):
                optimizer.zero_grad()
                mu, sigma = model.forward(x_train)
                loss = model.batch_loss(mu, sigma, y_train.view(-1, 1))
                loss.backward()
                optimizer.step()
        model.eval()
        mu, sigma = model.forward(torch.from_numpy(x_test).float().to('cuda:0'))
        logits = torch.sigmoid(mu).detach().cpu().numpy()
        logits_all.append(logits.reshape(-1))
        labels_all.append(y_test)
        print('Iter {}/10 done'.format(counter))
        counter += 1
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
