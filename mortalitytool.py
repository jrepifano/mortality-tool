import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from smoothInfluence import influence
from VDP_influence import VDP_influence
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import percentileofscore
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, normalize
from VDP_Layers import VDP_FullyConnected, VDP_Relu


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
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

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
    return x, y


class deterministic:
    def __init__(self):
        self.model = Model(197, 198, 112)
        self.model.load_state_dict(torch.load(os.getcwd()+'/saved_models/d.pt'))
        self.model.to('cuda:0')
        self.model.eval()
        self.imputer = np.load(os.getcwd()+'/saved_models/d_imputer.npy', allow_pickle=True).item()
        self.scaler = np.load(os.getcwd()+'/saved_models/d_scaler.npy', allow_pickle=True).item()
        self.feature_names = ['Lactate', 'Mech Vent', 'Eyes', 'Motor', 'Verbal', 'Albumin',
                              'Age', 'Creatinine', 'BUN', 'PT-INR', 'WBC', 'MAP']
        self.x, self.y = load_data()
        self.x = self.scaler.transform(self.imputer.transform(self.x))

    def inference(self, input_vec):
        if np.isnan(input_vec).any():
            input_vector = self.scaler.transform(self.imputer.transform(input_vec.reshape((1, -1))))
        else:
            input_vector = self.scaler.transform(input_vec.reshape((1, -1)))
        prediction = torch.sigmoid(self.model(torch.from_numpy(input_vector).float().to('cuda:0'))).detach().cpu().numpy()[0][0]
        output = {'prediction': [float("{:.4f}".format(float(prediction)))],
                  'values': [float("{:.4f}".format(float(i))) for i in
                             np.hstack(self.scaler.inverse_transform(input_vector)).tolist()]}
        return output

    def get_explanation(self, input_vec):
        if np.isnan(input_vec).any():
            input_vector = self.scaler.transform(self.imputer.transform(input_vec.reshape((1, -1))))
        else:
            input_vector = self.scaler.transform(input_vec.reshape((1, -1)))
        infl = influence(self.x, self.y, input_vector, np.array([1.]), self.model, self.model.lin4.weight)
        infl_normalized = normalize(infl).reshape(-1)
        self.plot_explanation(infl_normalized, input_vec)
        return infl_normalized

    def plot_explanation(self, influence, input_vec):
        plt.style.use("default")
        # influence = -1 * influence
        neg_val = np.where(np.flip(influence) < 0)[0]
        pos_val = np.where(np.flip(influence) > 0)[0]
        nans = np.where(np.isnan(np.flip(input_vec)))[0]
        fig, ax = plt.subplots()
        legend_elements = [Line2D([0], [0], color='seagreen', lw=4, label='Indicates Survival'),
                           Line2D([0], [0], color='firebrick', lw=4, label='Indicates Mortality'),
                           Line2D([0], [0], color='darkgrey', lw=4, label='Imputed (disregard)')]
        plot = ax.barh(np.flip(self.feature_names), np.flip(influence), align='center')
        ax.legend(handles=legend_elements, loc='lower left')
        [plot[value].set_color('firebrick') for value in neg_val]
        [plot[value].set_color('seagreen') for value in pos_val]
        [plot[value].set_color('darkgrey') for value in nans]
        plt.show()
        plt.style.use("dark_background")
        fig, ax = plt.subplots()
        legend_elements = [Line2D([0], [0], color='seagreen', lw=4, label='Indicates Survival'),
                           Line2D([0], [0], color='firebrick', lw=4, label='Indicates Mortality'),
                           Line2D([0], [0], color='darkgrey', lw=4, label='Imputed (disregard)')]
        plot = ax.barh(np.flip(self.feature_names), np.flip(influence), align='center')
        ax.legend(handles=legend_elements, loc='lower left')
        [plot[value].set_color('firebrick') for value in neg_val]
        [plot[value].set_color('seagreen') for value in pos_val]
        [plot[value].set_color('darkgrey') for value in nans]
        plt.show()


class stochastic:
    def __init__(self):
        self.model = VDPNet(31, 93, 94)
        self.model.load_state_dict(torch.load(os.getcwd() + '/saved_models/s.pt'))
        self.model.to('cuda:0')
        self.model.eval()
        self.imputer = np.load(os.getcwd() + '/saved_models/s_imputer.npy', allow_pickle=True).item()
        self.scaler = np.load(os.getcwd() + '/saved_models/s_scaler.npy', allow_pickle=True).item()
        self.diags = np.load(os.getcwd()+'/saved_models/s_train_diags.npy')
        self.feature_names = ['Lactate', 'Mech Vent', 'Eyes', 'Motor', 'Verbal', 'Albumin',
                              'Age', 'Creatinine', 'BUN', 'PT-INR', 'WBC', 'MAP']
        self.x, self.y = load_data()
        self.x = self.scaler.transform(self.imputer.transform(self.x))

    def inference(self, input_vec):
        if np.isnan(input_vec).any():
            input_vector = self.scaler.transform(self.imputer.transform(input_vec.reshape((1, -1))))
        else:
            input_vector = self.scaler.transform(input_vec.reshape((1, -1)))
        prediction, sigma = self.model.forward(torch.from_numpy(input_vector).float().to('cuda:0'))
        prediction = torch.sigmoid(prediction).detach().cpu().numpy()[0][0]
        sigma = np.mean(np.diagonal(sigma.detach().cpu().numpy()[0]))
        percentile = percentileofscore(self.diags, sigma)
        output = {'prediction':   [float("{:.4f}".format(float(prediction)))],
                  'values':       [float("{:.4f}".format(float(i))) for i in np.hstack(self.scaler.inverse_transform(input_vector)).tolist()],
                  'confidence': [float("{:.2f}".format(float(100-percentile)))]}
        return output

    def get_explanation(self, input_vec):
        if np.isnan(input_vec).any():
            input_vector = self.scaler.transform(self.imputer.transform(input_vec.reshape((1, -1))))
        else:
            input_vector = self.scaler.transform(input_vec.reshape((1, -1)))
        trainset = data_loader(self.x, self.y)
        btchsz = len(self.x)//15
        trainloader = DataLoader(trainset, batch_size=btchsz, shuffle=True, num_workers=2, generator=torch.Generator(device='cuda'))
        influences = []
        for itr, (batch_data, batch_labels) in enumerate(trainloader):
            infl = VDP_influence(batch_data, batch_labels, input_vector.reshape(1, -1), np.array([1.]), self.model, self.model.fullyCon4.mean_fc.weight)
            influences.append(infl)
            if itr % 1 == 0 or itr == int(len(self.x) / btchsz) - 1:
                print('Finished Batch :' + str(itr + 1) + '/' + str(int(len(self.x) / btchsz)+1))
        infl_normalized = np.sum(normalize(np.squeeze(influences)), axis=0)
        self.plot_explanation(infl_normalized, input_vec)
        return infl_normalized

    def plot_explanation(self, influence, input_vec):
        plt.style.use("default")
        influence = -1 * influence
        neg_val = np.where(np.flip(influence) < 0)[0]
        pos_val = np.where(np.flip(influence) > 0)[0]
        nans = np.where(np.isnan(np.flip(input_vec)))[0]
        fig, ax = plt.subplots()
        legend_elements = [Line2D([0], [0], color='seagreen', lw=4, label='Indicates Survival'),
                           Line2D([0], [0], color='firebrick', lw=4, label='Indicates Mortality'),
                           Line2D([0], [0], color='darkgrey', lw=4, label='Imputed (disregard)')]
        plot = ax.barh(np.flip(self.feature_names), np.flip(influence), align='center')
        ax.legend(handles=legend_elements, loc='lower left')
        [plot[value].set_color('firebrick') for value in neg_val]
        [plot[value].set_color('seagreen') for value in pos_val]
        [plot[value].set_color('darkgrey') for value in nans]
        plt.show()
        plt.style.use("dark_background")
        fig, ax = plt.subplots()
        legend_elements = [Line2D([0], [0], color='seagreen', lw=4, label='Indicates Survival'),
                           Line2D([0], [0], color='firebrick', lw=4, label='Indicates Mortality'),
                           Line2D([0], [0], color='darkgrey', lw=4, label='Imputed (disregard)')]
        plot = ax.barh(np.flip(self.feature_names), np.flip(influence), align='center')
        ax.legend(handles=legend_elements, loc='lower left')
        [plot[value].set_color('firebrick') for value in neg_val]
        [plot[value].set_color('seagreen') for value in pos_val]
        [plot[value].set_color('darkgrey') for value in nans]
        plt.show()
