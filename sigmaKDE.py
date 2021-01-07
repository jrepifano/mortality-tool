import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    preds = np.load(os.getcwd()+'/saved_models/s_train_preds.npy')
    diags = np.load(os.getcwd()+'/saved_models/s_train_diags.npy')
    y = np.load(os.getcwd()+'/saved_models/s_train_labels.npy')
    right = np.argwhere(np.round(preds) == y).reshape(-1)
    wrong = np.argwhere(np.round(preds) != y).reshape(-1)

    sigma_right = pd.DataFrame(diags[right], columns=['Right'])
    sigma_wrong = pd.DataFrame(diags[wrong], columns=['Wrong'])
    sigma_all = pd.DataFrame(diags, columns=['All'])

    # p1 = sns.kdeplot(sigma_all['All'], shade=True, color="b")
    p2 = sns.kdeplot(sigma_right['Right'], shade=True, color="g")
    p3 = sns.kdeplot(sigma_wrong['Wrong'], shade=True, color="r")
    plt.legend(['Classifier is Correct', 'Classifier is Incorrect'])
    plt.xlim([-2, 10])
    plt.xlabel('Mean Sigma')
    plt.show()


if __name__ == '__main__':
    main()
