import os
import torch
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from VDP_Layers import VDP_FullyConnected, VDP_Relu, VDP_Softmax
from torch.utils.data import DataLoader, Dataset


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
    x_eicu = pd.read_csv(cd+'/../data/x_eicu.csv')
    y_eicu = pd.read_csv(cd+'/../data/y_eicu.csv')
    mimic = pd.read_csv(cd + '/../data/mimic.csv')
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
    x, y = load_data()
    imputer = IterativeImputer()
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(x))
    x, y = RandomUnderSampler().fit_resample(x, y)
    trainset = data_loader(x, y)
    trainloader = DataLoader(trainset, batch_size=1000, shuffle=True)
    model = VDPNet(31, 93, 94)
    no_epochs = 18
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002219, weight_decay=0.006427, momentum=0.7589,
                                nesterov=True)
    model.train()
    model.to('cuda:0')
    for epoch in range(no_epochs):
        for itr, (x_batch, y_batch) in enumerate(trainloader):
            optimizer.zero_grad()
            mu, sigma = model.forward(x_batch)
            loss = model.batch_loss(mu, sigma, y_batch.view(-1, 1))
            loss.backward()
            optimizer.step()
    preds = []
    diags = []
    x, y = load_data()
    x = scaler.transform(imputer.transform(x))
    trainset = data_loader(x, y)
    trainloader = DataLoader(trainset, batch_size=1000, shuffle=False)
    for itr, (x_batch, y_batch) in enumerate(trainloader):
        mu, sigma = model.forward(x_batch)
        preds.append(torch.sigmoid(mu).detach().cpu().numpy())
        diags.append([np.mean(np.diagonal(matrix.detach().cpu().numpy())) for matrix in sigma])
    torch.save(model.state_dict(), os.getcwd()+'/saved_models/s.pt')
    np.save(os.getcwd()+'/../saved_models/s_imputer', imputer, allow_pickle=True)
    np.save(os.getcwd()+'/../saved_models/s_scaler', scaler, allow_pickle=True)
    np.save(os.getcwd()+'/../saved_models/s_train_labels', y)
    np.save(os.getcwd()+'/../saved_models/s_train_preds', np.vstack(preds).reshape(-1))
    np.save(os.getcwd()+'/../saved_models/s_train_diags', np.hstack(diags))


if __name__ == '__main__':
    main()
