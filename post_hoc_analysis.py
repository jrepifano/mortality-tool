import json
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict, ChainMap, Counter
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import ks_2samp, chisquare, percentileofscore


def calc_metrics(results):
    vdp_score = roc_auc_score(results['health'], results['vdp_pred'])
    det_score = roc_auc_score(results['health'], results['det_pred'])
    print(f'VDP ROC AUC: {vdp_score:.2f}')
    print(f'DET ROC AUC: {det_score:.2f}')

    vdp_score = accuracy_score(results['health'], results['vdp_pred'].round())
    det_score = accuracy_score(results['health'], results['det_pred'].round())
    clinician_score = accuracy_score(results.dropna()['health'], results.dropna()['health_predicted'])
    print(f'VDP Accuracy: {vdp_score:.2f}')
    print(f'DET Accuracy: {det_score:.2f}')
    print(f'Clinician Accuracy: {clinician_score:.2f}')
    
    
def load_data():
    x_eicu = pd.read_csv('data/x_eicu.csv')
    y_eicu = pd.read_csv('data/y_eicu.csv')
    mimic = pd.read_csv('data/mimic.csv')
    assert np.all(x_eicu['patientunitstayid'].to_numpy() == y_eicu['patientunitstayid'].to_numpy())
    feature_list = ['lactate', 'oobventday1', 'eyes', 'motor', 'verbal', 'albumin_x',
                    'age', 'creatinine_x', 'BUN', 'PT - INR', 'WBC x 1000', 'meanbp']
    feature_list_mimic = ['Lactate', 'firstdayvent', 'gcseyes', 'gcsmotor', 'gcsverbal', 'Albumin',
                          'Age', 'Creatinine', 'BUN', 'INR', 'WBC', 'MAP']
    feature_list_app = ['lactate', 'firstdayvent', 'gcseyes', 'gcsmotor', 'gcsverbal', 'albumin', 'age', 'creatinine', 'bun', 'inr', 'wbc', 'map']
    x_eicu = x_eicu[feature_list].to_numpy()
    y_eicu = y_eicu['actualicumortality'].to_numpy()
    x_mimic = mimic[feature_list_mimic].to_numpy()
    y_mimic = mimic['Mortality'].to_numpy()
    x = np.vstack((x_eicu, x_mimic))
    y = np.hstack((y_eicu, y_mimic))
    return pd.DataFrame(x, columns=feature_list_app), y


def classification_results():
    input_data = pd.read_csv('app_outputs_both_cohorts_5-17/_DATABASE_DUMP_20230517__table_inputs.csv')
    output_data = pd.read_csv('app_outputs_both_cohorts_5-17/_DATABASE_DUMP_20230517__table_outputs.csv')

    print(f'Number of input samples {len(input_data)}')
    unique_ids = [elem for elem in output_data['patientID'].unique() if 'test' not in elem.lower()]
    unique_ids = [elem for elem in unique_ids if 'coop' not in elem.lower()]
    print(f'Number of non-test entries: {len(unique_ids)}')
    pred_conv = {0: np.nan, 1:0, 2:1}
    results = list()
    for id in unique_ids:
        filt = (output_data['patientID'] == id) & (output_data['health'] != 0)
        filtered_df = output_data.loc[filt]
        if len(filtered_df) == 0:
            continue
        vdp_pred = json.loads(filtered_df.loc[filtered_df['model'] == 'A']['data'].item())['prediction'][0]
        vdp_conf = json.loads(filtered_df.loc[filtered_df['model'] == 'A']['data'].item())['confidence'][0]
        det_pred = json.loads(filtered_df.loc[filtered_df['model'] == 'B']['data'].item())['prediction'][0]

        results.append({
            'PatientID': id,
            'date': filtered_df['date'].iloc[0],
            'vdp_pred': vdp_pred,
            'vdp_conf': vdp_conf,
            'det_pred': det_pred,
            'health_predicted': pred_conv[filtered_df['health_predicted'].max()],
            'health': pred_conv[filtered_df['health'].max()]
        })
    print('Combined Cohort Results')
    results = pd.DataFrame(results)
    results['date'] = pd.to_datetime(results['date'])
    calc_metrics(results)
    print('Covid cohort metrics')
    calc_metrics(results.query('date < 2023'))
    print(len(results.query('date < 2023')))
    print('normal cohort metrics')
    calc_metrics(results.query('date > 2023'))
    print(len(results.query('date > 2023')))
    

def load_all_data():
    output_data = pd.read_csv('app_outputs_both_cohorts_5-17/_DATABASE_DUMP_20230517__table_outputs.csv')

    full_feats = ['lactate', 'firstdayvent', 'gcseyes', 'gcsmotor', 'gcsverbal', 'albumin', 'age', 'bun', 'inr', 'wbc', 'map', 'creatinine']

    unique_ids = [elem for elem in output_data['patientID'].unique() if 'test' not in elem.lower()]
    unique_ids = [elem for elem in unique_ids if 'coop' not in elem.lower()]
    results = list()
    imputed_data = list()
    for id in unique_ids:
        filt = (output_data['patientID'] == id)
        filtered_df = output_data.loc[filt]
        if len(filtered_df) == 0:
            continue
        data_a = filtered_df.loc[filtered_df['model'] == 'A']
        data_b = filtered_df.loc[filtered_df['model'] == 'B']
        if len(data_a) > 1:
            data_a = json.loads(data_a.iloc[0]['data'])
            data_b = json.loads(data_b.iloc[0]['data'])
        else:
            data_a = json.loads(data_a['data'].item())
            data_b = json.loads(data_b['data'].item())
        imputed = defaultdict(lambda : 0)
        [imputed[feat] for feat in full_feats]
        [imputed.update({feat:1}) for feat in data_a['imputed']]
        subj_data = {'subjectID': id, 'date': filtered_df['date'].iloc[0]}
        imputed_data.append(dict(ChainMap(dict(imputed), subj_data)))
        feat = {col: value for col, value in zip(full_feats, data_a['values'])}
        predictions = {'Model A': data_a['prediction'][0], 'Confidence': data_a['confidence'][0], 'Model B': data_b['prediction'][0], 'Outcome': filtered_df.iloc[0]['health']}
        results.append(dict(ChainMap(feat, predictions, subj_data)))
    results = pd.DataFrame(results)
    results['date'] = pd.to_datetime(results['date'])
    imputed = pd.DataFrame(imputed_data)
    X_train, y_train = load_data()
    return X_train, y_train, results, imputed


def feature_analysis():
    continuous_feats = ['lactate', 'albumin', 'creatinine', 'bun', 'inr', 'wbc', 'map']
    X_train, y_train, results, imputed = load_all_data()
    stats = list()
    for feat in continuous_feats:
        results[feat][imputed[feat]==1] = np.nan
        train_feat = X_train[feat].dropna()
        cohort_1 = results.query('date < 2023')[feat].dropna()
        cohort_2 = results.query('date > 2023')[feat].dropna()
        cohort_1_p_value = ks_2samp(train_feat, cohort_1)[1]
        cohort_2_p_value = ks_2samp(train_feat, cohort_2)[1]
        stats.append({
            'feat': feat,
            'cohort_1': cohort_1_p_value,
            'cohort_2': cohort_2_p_value
        })
        if cohort_1_p_value < 0.01 and cohort_2_p_value > 0.01:
            sns.set(font_scale = 2)
            plt.figure(figsize=(18, 10))
            sns.kdeplot(train_feat, fill=True, label='Training Set', cut=0)
            sns.kdeplot(cohort_1, fill=True, label='2021 Cohort', cut=0)
            sns.kdeplot(cohort_2, fill=True, label='2023 Cohort', cut=0)
            plt.legend()
            if feat == 'lactate':
                plt.xlabel(feat.title()+' (mmol/L)')
                plt.xlim([0, 15])
            elif feat == 'albumin':
                plt.xlabel(feat.title()+' (g/dl)')
                plt.xlim([0, 10])
            plt.savefig(f'results/{feat}.png')
            plt.clf()
    stats = pd.DataFrame(stats)
    health = list()
    alive, expired = np.unique(y_train, return_counts=True)[1] / len(y_train)
    health.append({'Cohort': 'Training Set', 'Proporation with Expired Outcome': expired})
    cohort_1_health = results.query('date < 2023')['Outcome']
    cohort_1_health = cohort_1_health.loc[cohort_1_health != 0].values -1
    alive, expired = np.unique(cohort_1_health, return_counts=True)[1] / len(cohort_1_health)
    health.append({'Cohort': '2021', 'Proporation with Expired Outcome': expired})
    cohort_2_health = results.query('date > 2023')['Outcome']
    cohort_2_health = cohort_2_health.loc[cohort_2_health != 0].values -1
    alive, expired = np.unique(cohort_2_health, return_counts=True)[1] / len(cohort_2_health)
    health.append({'Cohort': '2023', 'Proporation with Expired Outcome': expired})
    plt.figure(figsize=(18, 10))
    sns.barplot(data=pd.DataFrame(health), x='Cohort', y='Proporation with Expired Outcome')
    plt.savefig('results/outcome_prop.png')
    plt.clf()
    print(health)
    chi2_cohort_1 = chisquare(np.unique(cohort_1_health, return_counts=True)[1] / len(cohort_1_health), np.unique(y_train, return_counts=True)[1] / len(y_train))
    print(f'cohort 1 chisquare label: {chi2_cohort_1}')
    chi2_cohort_2 = chisquare(np.unique(cohort_2_health, return_counts=True)[1] / len(cohort_2_health), np.unique(y_train, return_counts=True)[1] / len(y_train))
    print(f'cohort 2 chisquare label: {chi2_cohort_2}')
    pass


def uncertainty_analysis():
    X_train, y_train, results, imputed = load_all_data()
    y_diags = np.load('saved_models/s_train_diags.npy')
    y_preds = np.load('saved_models/s_train_preds.npy')
    
    train_confidence = list()
    for sigma in tqdm(y_diags):
        percentile = percentileofscore(y_diags, sigma)
        train_confidence.append(round(100 - percentile, 2))
    sns.set(font_scale = 2)
    plt.figure(figsize=(18, 10))
    sns.kdeplot(train_confidence, fill=True, label='Training Set', cut=0)
    sns.kdeplot(results.query('date < 2023')['Confidence'], fill=True, label='2021 Cohort', cut=0)
    sns.kdeplot(results.query('date > 2023')['Confidence'], fill=True, label='2023 Cohort', cut=0)
    plt.xlabel('Model Certainty')
    plt.legend()
    plt.savefig('results/cohort_uncertainty.png')
    plt.clf()
    print('2021 Cohort KS test')
    print(ks_2samp(np.array(train_confidence), results.query('date < 2023')['Confidence'].values))
    print('2023 Cohort KS test')
    print(ks_2samp(np.array(train_confidence), results.query('date > 2023')['Confidence'].values))
    pass
    train_conf_right = np.array(train_confidence)[y_preds.round() == y_train]
    train_conf_wrong = np.array(train_confidence)[~(y_preds.round() == y_train)]
    plt.figure(figsize=(18, 10))
    sns.kdeplot(train_conf_right, fill=True, cut=0, label='Classifer is Correct', color=sns.color_palette()[2])
    sns.kdeplot(train_conf_wrong, fill=True, cut=0, label='Classifer is Incorrect', color=sns.color_palette()[3])
    plt.xlabel('Model Certainty')
    plt.legend()
    plt.savefig('results/train_uncertainty_kde.png')
    plt.clf()
    train_conf_right = np.array(y_diags)[y_preds.round() == y_train]
    train_conf_wrong = np.array(y_diags)[~(y_preds.round() == y_train)]
    plt.figure(figsize=(18, 10))
    sns.kdeplot(train_conf_right, fill=True, cut=0, label='Classifer is Correct', color=sns.color_palette()[2])
    sns.kdeplot(train_conf_wrong, fill=True, cut=0, label='Classifer is Incorrect', color=sns.color_palette()[3])
    plt.xlabel('Output Variance')
    plt.xlim([0, 15])
    plt.legend()
    plt.savefig('results/train_sigma_kde.png')
    plt.clf()
    pass
    
    
def explanation_analysis():
    output_data = pd.read_csv('app_outputs_both_cohorts_5-17/_DATABASE_DUMP_20230517__table_outputs.csv')

    full_feats = ['lactate', 'firstdayvent', 'gcseyes', 'gcsmotor', 'gcsverbal', 'albumin', 'age', 'bun', 'inr', 'wbc', 'map', 'creatinine']

    unique_ids = [elem for elem in output_data['patientID'].unique() if 'test' not in elem.lower()]
    unique_ids = [elem for elem in unique_ids if 'coop' not in elem.lower()]
    explanations = list()
    for id in unique_ids:
        filt = (output_data['patientID'] == id)
        filtered_df = output_data.loc[filt]
        if len(filtered_df) == 0:
            continue
        data_a = filtered_df.loc[filtered_df['model'] == 'A']
        data_b = filtered_df.loc[filtered_df['model'] == 'B']
        if len(data_a) > 1:
            data_a = json.loads(data_a.iloc[0]['data'])
            data_b = json.loads(data_b.iloc[0]['data'])
        else:
            data_a = json.loads(data_a['data'].item())
            data_b = json.loads(data_b['data'].item())
        imputed = defaultdict(lambda : 0)
        [imputed[feat] for feat in full_feats]
        [imputed.update({feat:1}) for feat in data_a['imputed']]
        vdp_exp = data_a['explanation']
        det_exp = data_b['explanation']
        explanations.append({
        'subjectID': id,
        'date': filtered_df['date'].iloc[0],
        'imputed': list(imputed.values()),
        'vdp_exp': vdp_exp,
        'det_exp': det_exp
        })
    explanations = pd.DataFrame(explanations)
    explanations['date'] = pd.to_datetime(explanations['date'])
    vdp_exp_2021 = np.vstack(explanations.query('date < 2023')['vdp_exp'].to_numpy())
    vdp_exp_2023 = np.vstack(explanations.query('date > 2023')['vdp_exp'].to_numpy())
    
    det_exp_2021 = np.vstack(explanations.query('date < 2023')['det_exp'].to_numpy())
    det_exp_2023 = np.vstack(explanations.query('date > 2023')['det_exp'].to_numpy())
    
    vdp_top_3_2021 = list()
    det_top_3_2021 = list()
    
    vdp_top_3_2023 = list()
    det_top_3_2023 = list()
    for vdp, det in zip(vdp_exp_2021, det_exp_2021):
        vdp_top_3_2021 += list(np.array(full_feats)[np.argsort(np.abs(vdp))[::-1]][:3])
        det_top_3_2021 += list(np.array(full_feats)[np.argsort(np.abs(det))[::-1]][:3])
    for vdp, det in zip(vdp_exp_2023, det_exp_2023):
        vdp_top_3_2023 += list(np.array(full_feats)[np.argsort(np.abs(vdp))[::-1]][:3])
        det_top_3_2023 += list(np.array(full_feats)[np.argsort(np.abs(det))[::-1]][:3])
    print(f'VDP EXP 2021: {Counter(vdp_top_3_2021)}')
    print(f'VDP EXP 2023: {Counter(vdp_top_3_2023)}')

    print(f'DET EXP 2021: {Counter(det_top_3_2021)}')
    print(f'DET EXP 2023: {Counter(det_top_3_2023)}')
    
        
    sign_vdp_2021 = np.sum(np.sign(vdp_exp_2021), axis=0)# / len(vdp_exp_2021)
    sign_vdp_2023 = np.sum(np.sign(vdp_exp_2023), axis=0)# / len(vdp_exp_2023)
    
    sign_det_2021 = np.sum(np.sign(det_exp_2021), axis=0)# / len(det_exp_2021)
    sign_det_2023 = np.sum(np.sign(det_exp_2023), axis=0)# / len(det_exp_2023)
    
    print(f'VDP Sign 2021: \n{sign_vdp_2021}')
    print(f'VDP Sign 2023: \n{sign_vdp_2023}')
    
    print(f'DET Sign 2021: \n{sign_det_2021}')
    print(f'DET Sign 2023: \n{sign_det_2023}')
    pass


if __name__ == '__main__':
    # classification_results()
    feature_analysis()
    # uncertainty_analysis()
    # explanation_analysis()

    