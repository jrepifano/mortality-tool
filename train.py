import os
import torch
import numpy as np
from numpy.random import randn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, balanced_accuracy_score


def add_noise(s, snr):
    var_s = np.var(s, axis=1)
    var_n = var_s / (10 ** (snr / 10))
    rand_arr = randn(s.shape[0], s.shape[1])
    n = np.sqrt(var_n).reshape((-1, 1)) * rand_arr
    return s + n


def load_data(split=False, resample=False):
    dir = os.getcwd()
    X_eicu = np.load(dir + '/data/X_eicu.npy')
    y_eicu = np.load(dir + '/data/y_eicu.npy')
    X_mimc = np.load(dir + '/data/X_mimic.npy')
    y_mimic = np.load(dir + '/data/y_mimic.npy')
    X = np.vstack((X_eicu, X_mimc))
    y = np.hstack((y_eicu, y_mimic))
    imputer = IterativeImputer()
    scaler = StandardScaler()
    if split == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)
        X_train = imputer.fit_transform(X_train)
        X_train = scaler.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        X_test = scaler.transform(X_test)
        if resample == True:
            X_train, y_train = RandomUnderSampler(sampling_strategy=1).fit_resample(X_train, y_train)
            print('Shape after resampling: ' + str(X_train.shape))
        return X_train, X_test, y_train, y_test
    else:
        X = imputer.fit_transform(X)
        X = scaler.fit_transform(X)
        if resample == True:
            X, y = RandomUnderSampler().fit_resample(X, y)
        return X, y


def create_model(num_layers):
    arch = []
    out_feats = 10
    for layer in range(num_layers):
        in_feats = out_feats if layer != 0 else 10
        out_feats = 1000 // num_layers if layer != num_layers - 1 else 1
        arch.append(torch.nn.Linear(in_feats, out_feats))
        arch.append(torch.nn.SELU() if layer != num_layers - 1 else torch.nn.Identity())
    model = torch.nn.Sequential(*arch)
    print(model)
    return model


def main():
    np.random.seed(1234567890)
    torch.manual_seed(1234567890)
    split = True
    resample = False
    snr = 10
    if split == True:
        X_train, X_test, y_train, y_test = load_data(split, resample)
        y_test_true = y_test
        X_test, y_test = torch.from_numpy(X_test).float().to('cuda:0'), torch.from_numpy(y_test).float().to('cuda:0')
    else:
        X_train, y_train = load_data(split, resample)
    X_train_noise = torch.from_numpy(add_noise(X_train, snr)).float().to('cuda:0')
    y_train_true = y_train
    X_train, y_train = torch.from_numpy(X_train).float().to('cuda:0'), torch.from_numpy(y_train).float().to(
        'cuda:0')
    model = create_model(3)
    model.to('cuda:0')
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([12], device='cuda:0'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100
    print_iter = 20
    model.train()
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    train_precision = []
    test_precision = []
    train_sensitivity = []
    test_sensitivity = []
    train_specificity = []
    test_specificity = []
    train_roc_auc = []
    test_roc_auc = []
    train_prc_auc = []
    test_prc_auc = []
    train_balanced_acc = []
    test_balanced_acc = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model.forward(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        # if epoch == 0 or (epoch + 1) % print_iter == 0 or epoch == num_epochs - 1:
        logits = torch.sigmoid(outputs).detach().cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(y_train_true, np.round(logits)).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        roc_auc = roc_auc_score(y_train_true, logits)
        prc_auc = average_precision_score(y_train_true, logits)
        balanced_acc = balanced_accuracy_score(y_train_true, np.round(logits))
        train_loss.append(loss.item())
        train_acc.append(accuracy)
        train_precision.append(precision)
        train_sensitivity.append(recall)
        train_specificity.append(specificity)
        train_roc_auc.append(roc_auc)
        train_prc_auc.append(prc_auc)
        train_balanced_acc.append(balanced_acc)
        # print('\n------------------------TRAINING DATA------------------------')
        # print('Epoch: {}/{}, Train Loss: {:.8f}'.format(epoch + 1, num_epochs, loss))
        # print('Acc: {:.2f}, Precision: {:.2f}, Sensitivity: {:.2f}, Specificity: {:.2f}'.format(
        #     accuracy, precision, recall, specificity))
        # print('Balanced Acc: {:.2f}, ROC AUC: {:.2f}, PRC AUC: {:.2f}'.format(balanced_acc, roc_auc, prc_auc))
        # print('Un-Normalized Confusion Matrix')
        # print('           Predicted 0 Predicted 1')
        # print('True Negative {:.0f}  {:.0f}'.format(tn, fp))
        # print('True Positive {:.0f}  {:.0f}'.format(fn, tp))
        # print('Normalized Confusion Matrix')
        # print('           Predicted 0 Predicted 1')
        # print('True Negative {:.2f}  {:.2f}'.format(tn / (tn + fp), fp / (tn + fp)))
        # print('True Positive {:.2f}  {:.2f}'.format(fn / (fn + tp), tp / (fn + tp)))
        # model.eval()
        # outputs = model.forward(X_train_noise)
        # loss = criterion(outputs, y_train.view(-1, 1))
        # if epoch == 0 or (epoch + 1) % print_iter == 0 or epoch == num_epochs - 1:
        #     logits = torch.sigmoid(outputs).detach().cpu().numpy()
        #     tn, fp, fn, tp = confusion_matrix(y_train_true, np.round(logits)).ravel()
        #     accuracy = (tp + tn) / (tp + tn + fp + fn)
        #     precision = tp / (tp + fp)
        #     recall = tp / (tp + fn)
        #     specificity = tn / (tn + fp)
        #     roc_auc = roc_auc_score(y_train_true, logits)
        #     prc_auc = average_precision_score(y_train_true, logits)
        #     balanced_acc = balanced_accuracy_score(y_train_true, np.round(logits))
        #     print('-----------Training Data with noise SNR {}-----------'.format(snr))
        #     print('Acc: {:.2f}, Precision: {:.2f}, Sensitivity: {:.2f}, Specificity: {:.2f}'.format(
        #         accuracy, precision, recall, specificity))
        #     print('Balanced Acc: {:.2f}, ROC AUC: {:.2f}, PRC AUC: {:.2f}'.format(balanced_acc, roc_auc, prc_auc))
        if split == True:
            model.eval()
            outputs = model.forward(X_test)
            loss = criterion(outputs, y_test.view(-1, 1))
            logits = torch.sigmoid(outputs).detach().cpu().numpy()
            tn, fp, fn, tp = confusion_matrix(y_test_true, np.round(logits)).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            specificity = tn / (tn + fp)
            roc_auc = roc_auc_score(y_test_true, logits)
            prc_auc = average_precision_score(y_test_true, logits)
            balanced_acc = balanced_accuracy_score(y_test_true, np.round(logits))
            # print('------------------------TEST DATA------------------------')
            # print('Test Loss after {} epochs: {:.8f}'.format(num_epochs, loss))
            # print('Test Acc: {:.2f}, Test Precision: {:.2f}, Test Sensitivity: {:.2f}, Test Specificity: {:.2f}'.format(
            #     accuracy, precision, recall, specificity))
            # print('Test Balanced Acc: {:.2f}, Test ROC AUC: {:.2f}, Test PRC AUC: {:.2f}'.format(balanced_acc, roc_auc, prc_auc))
            # print('Un-Normalized Confusion Matrix')
            # print('           Predicted 0 Predicted 1')
            # print('True Negative {:.0f}  {:.0f}'.format(tn, fp))
            # print('True Positive {:.0f}  {:.0f}'.format(fn, tp))
            # print('Normalized Confusion Matrix')
            # print('           Predicted 0 Predicted 1')
            # print('True Negative {:.2f}  {:.2f}'.format(tn/(tn+fp), fp/(tn+fp)))
            # print('True Positive {:.2f}  {:.2f}'.format(fn/(fn+tp), tp/(fn+tp)))
            test_loss.append(loss.item())
            test_acc.append(accuracy)
            test_precision.append(precision)
            test_sensitivity.append(recall)
            test_specificity.append(specificity)
            test_roc_auc.append(roc_auc)
            test_prc_auc.append(prc_auc)
            test_balanced_acc.append(balanced_acc)
    metrics = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'roc_auc', 'prc_auc', 'balanced_acc']
    sns.lineplot(x=range(num_epochs), y=train_loss, label='train loss')
    sns.lineplot(x=range(num_epochs), y=test_loss, label='test loss')
    plt.xlabel(['Epoch'])
    plt.ylabel(['Loss'])
    plt.legend()
    plt.savefig(os.getcwd() + '/results/no_resampling/' + metrics[0])
    plt.clf()
    plt.cla()
    plt.close()
    sns.lineplot(x=range(num_epochs), y=train_acc, label='train acc')
    sns.lineplot(x=range(num_epochs), y=test_acc, label='test acc')
    plt.xlabel(['Epoch'])
    plt.ylabel(['Accuracy'])
    plt.legend()
    plt.savefig(os.getcwd() + '/results/no_resampling/' + metrics[1])
    plt.clf()
    plt.cla()
    plt.close()
    sns.lineplot(x=range(num_epochs), y=train_precision, label='train precision')
    sns.lineplot(x=range(num_epochs), y=test_precision, label='test precision')
    plt.xlabel(['Epoch'])
    plt.ylabel(['Precision'])
    plt.legend()
    plt.savefig(os.getcwd() + '/results/no_resampling/' + metrics[2])
    plt.clf()
    plt.cla()
    plt.close()
    sns.lineplot(x=range(num_epochs), y=train_sensitivity, label='train sensitivity')
    sns.lineplot(x=range(num_epochs), y=test_sensitivity, label='test sensitivity')
    plt.xlabel(['Epoch'])
    plt.ylabel(['Sensitivity'])
    plt.legend()
    plt.savefig(os.getcwd() + '/results/no_resampling/' + metrics[3])
    plt.clf()
    plt.cla()
    plt.close()
    sns.lineplot(x=range(num_epochs), y=train_specificity, label='train specificity')
    sns.lineplot(x=range(num_epochs), y=test_sensitivity, label='test specificity')
    plt.xlabel(['Epoch'])
    plt.ylabel(['Specificity'])
    plt.legend()
    plt.savefig(os.getcwd() + '/results/no_resampling/' + metrics[4])
    plt.clf()
    plt.cla()
    plt.close()
    sns.lineplot(x=range(num_epochs), y=train_roc_auc, label='train roc auc')
    sns.lineplot(x=range(num_epochs), y=test_roc_auc, label='test roc auc')
    plt.xlabel(['Epoch'])
    plt.ylabel(['ROC AUC'])
    plt.legend()
    plt.savefig(os.getcwd() + '/results/no_resampling/' + metrics[5])
    plt.clf()
    plt.cla()
    plt.close()
    sns.lineplot(x=range(num_epochs), y=train_prc_auc, label='train prc auc')
    sns.lineplot(x=range(num_epochs), y=test_prc_auc, label='test prc auc')
    plt.xlabel(['Epoch'])
    plt.ylabel(['PRC AUC'])
    plt.legend()
    plt.savefig(os.getcwd() + '/results/no_resampling/' + metrics[6])
    plt.clf()
    plt.cla()
    plt.close()
    sns.lineplot(x=range(num_epochs), y=train_balanced_acc, label='train balanced acc')
    sns.lineplot(x=range(num_epochs), y=test_balanced_acc, label='test balanced acc')
    plt.xlabel(['Epoch'])
    plt.ylabel(['Balanced Accuracy'])
    plt.legend()
    plt.savefig(os.getcwd() + '/results/no_resampling/' + metrics[7])
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == '__main__':
    main()
