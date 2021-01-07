import os
import torch
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


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


def main():
    x, y = load_data()
    imputer = IterativeImputer()
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(x))
    x, y = torch.from_numpy(x).float().to('cuda:0'), torch.from_numpy(y).float().to('cuda:0')
    model = Model(197, 198, 112)
    no_epochs = 127
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([14.80], device='cuda:0'))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03104, weight_decay=0.01043, momentum=0.4204,
                                nesterov=True)
    model.train()
    model.to('cuda:0')
    for epoch in range(no_epochs):
        optimizer.zero_grad()
        outputs = model.forward(x)
        loss = criterion(outputs, y.view(-1, 1))
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), os.getcwd()+'/saved_models/d.pt')
    np.save(os.getcwd()+'/saved_models/d_imputer', imputer, allow_pickle=True)
    np.save(os.getcwd()+'/saved_models/d_scaler', scaler, allow_pickle=True)


if __name__ == '__main__':
    main()
