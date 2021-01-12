import os
import torch
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score


def load_data():
    cd = os.getcwd()
    mimic = pd.read_csv(cd + '/data/mimic.csv')
    feature_list_mimic = ['Lactate', 'firstdayvent', 'gcseyes', 'gcsmotor', 'gcsverbal', 'Albumin',
                          'Age', 'Creatinine', 'BUN', 'INR', 'WBC', 'MAP']
    x_mimic = mimic[feature_list_mimic].to_numpy()
    y_mimic = mimic['Mortality'].to_numpy()
    return x_mimic, y_mimic


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


def main():
    x, y = load_data()
    model = Model(44, 79, 179)
    model.load_state_dict(torch.load(os.getcwd() + '/saved_models/d.pt'))
    model.to('cuda:0')
    model.eval()
    imputer = np.load(os.getcwd() + '/saved_models/d_imputer.npy', allow_pickle=True).item()
    scaler = np.load(os.getcwd() + '/saved_models/d_scaler.npy', allow_pickle=True).item()
    x = scaler.transform(imputer.transform(x))
    prediction = torch.sigmoid(model(torch.from_numpy(x).float().to('cuda:0'))).detach().cpu().numpy()
    post_test_plus = ((prediction / (1 - prediction)) * 4.382) / (((prediction / (1 - prediction)) * 4.382) + 1)
    post_test_minus = ((prediction / (1 - prediction)) * 0.2221) / (((prediction / (1 - prediction)) * 0.2221) + 1)
    adjusted_prediction = np.zeros_like(prediction)
    adjusted_prediction = post_test_plus
    # for i in range(len(adjusted_prediction)):
    #     if prediction[i] > 0.5:
    #         adjusted_prediction[i] = post_test_plus[i]
    #     else:
    #         adjusted_prediction[i] = post_test_minus[i]

    tn, fp, fn, tp = confusion_matrix(y, np.round(prediction)).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    roc_auc = roc_auc_score(y, prediction)
    prc_auc = average_precision_score(y, prediction)
    balanced_acc = balanced_accuracy_score(y, np.round(prediction))
    print('------------------------TEST DATA - Raw Prediction------------------------')
    print('Test Acc: {:.2f}, Test Precision: {:.2f}, Test Sensitivity: {:.2f}, Test Specificity: {:.2f}'.format(
        accuracy, precision, recall, specificity))
    print(
        'Test Balanced Acc: {:.2f}, Test ROC AUC: {:.2f}, Test PRC AUC: {:.2f}'.format(balanced_acc, roc_auc, prc_auc))
    print('Un-Normalized Confusion Matrix')
    print('           Predicted 0 Predicted 1')
    print('True Negative {:.0f}  {:.0f}'.format(tn, fp))
    print('True Positive {:.0f}  {:.0f}'.format(fn, tp))
    print('Normalized Confusion Matrix')
    print('           Predicted 0 Predicted 1')
    print('True Negative {:.2f}  {:.2f}'.format(tn / (tn + fp), fp / (tn + fp)))
    print('True Positive {:.2f}  {:.2f}'.format(fn / (fn + tp), tp / (fn + tp)))

    tn, fp, fn, tp = confusion_matrix(y, np.round(adjusted_prediction)).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    roc_auc = roc_auc_score(y, adjusted_prediction)
    prc_auc = average_precision_score(y, adjusted_prediction)
    balanced_acc = balanced_accuracy_score(y, np.round(adjusted_prediction))
    print('------------------------TEST DATA - Adjusted Prediction------------------------')
    print('Test Acc: {:.2f}, Test Precision: {:.2f}, Test Sensitivity: {:.2f}, Test Specificity: {:.2f}'.format(
        accuracy, precision, recall, specificity))
    print(
        'Test Balanced Acc: {:.2f}, Test ROC AUC: {:.2f}, Test PRC AUC: {:.2f}'.format(balanced_acc, roc_auc, prc_auc))
    print('Un-Normalized Confusion Matrix')
    print('           Predicted 0 Predicted 1')
    print('True Negative {:.0f}  {:.0f}'.format(tn, fp))
    print('True Positive {:.0f}  {:.0f}'.format(fn, tp))
    print('Normalized Confusion Matrix')
    print('           Predicted 0 Predicted 1')
    print('True Negative {:.2f}  {:.2f}'.format(tn / (tn + fp), fp / (tn + fp)))
    print('True Positive {:.2f}  {:.2f}'.format(fn / (fn + tp), tp / (fn + tp)))
    pass


if __name__ == '__main__':
    main()
