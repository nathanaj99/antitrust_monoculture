from folktables import ACSDataSource, ACSEmployment, ACSIncome
import pandas as pd
import numpy as np
import scipy
from sympy import *
import os
import seaborn as sns
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm

H, L, sigma = symbols('H L sigma')
# n = 200000
# test data -- this is a constant

test_state = 'IL'
year = '2018'

test_data = pd.read_csv(f'../empirical/data/{test_state}_{year}_1yr.csv')
n_test = len(test_data)
features_test, label_test, _ = ACSIncome.df_to_numpy(test_data)

def perturb_new_data1(features_inj, label_inj, proportion_perturb):
    rng = np.random.default_rng(12345)

    perturb = rng.random(len(label_inj)) > (1-proportion_perturb)

    label_inj_perturb = []
    for lab, per in zip(label_inj, perturb):
        if per:
            label_inj_perturb.append(not lab)
        else:
            label_inj_perturb.append(lab)

    label_inj_perturb = np.array(label_inj_perturb)
    
    return features_inj, label_inj_perturb

def generate_utils(preds1, preds2):    
    util1 = 0
    util2 = 0

    for idx, res in enumerate(label_test):
        model1 = preds1[idx]
        model2 = preds2[idx]

        if res: # consumer is high type
            if model1 and model2: # then they split the high price
                util1 += H/2
                util2 += H/2
            elif model1 and not model2:
                util1 += sigma*H
                util2 += (1-sigma)*L
            elif not model1 and model2:
                util2 += sigma*H
                util1 += (1-sigma)*L
            else:
                util1 += L/2
                util2 += L/2
        else: # consumer is low type
            if model1 and model2: # then they both get 0
                pass
            elif model1 and not model2:
                util2 += L
            elif not model1 and model2:
                util1 += L
            else:
                util1 += L/2
                util2 += L/2
                
    return util1, util2

def run_independent_n(state, year, lr, n, random_state, verbose=False):
    acs_data = pd.read_csv(f'data/{state}_{year}_1yr.csv')
    
    acs_data = acs_data.sample(n=n, random_state=random_state)
    features, label, group = ACSIncome.df_to_numpy(acs_data)

#     scaler = StandardScaler()
#     scaler.fit(features)

#     X_train_scaled = scaler.transform(features)
#     X_test_scaled = scaler.transform(features_test)

    lr.fit(features, label)
    preds = lr.predict(features_test)
    preds_proba = lr.predict_proba(features_test)
    
    tn, fp, fn, tp = confusion_matrix(label_test, preds, labels=[0,1]).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    tnr = tn / (tn + fp)
    if verbose:
        print(f'Accuracy: {accuracy_score(label_test, preds)}')
        print(f'Precision: {precision}')
        print(f'Recall/TPR: {recall}')
        print(f'TNR: {tnr}')
        print(f'AUC: {roc_auc_score(label_test, preds_proba[:, 1])}')
    
    return lr, preds, {'acc': accuracy_score(label_test, preds), 'tpr': recall, 'tnr': tnr}

def run_combined_n(state, year, lr, acs_data_inj, n, random_state, perturb=None, frac=1, verbose=False):
    acs_data = pd.read_csv(f'data/{state}_{year}_1yr.csv')
    
    acs_data = acs_data.sample(n=n, random_state=random_state)
    acs_data_inj = acs_data_inj.sample(n=n, random_state=random_state)
    
    acs_data = acs_data.sample(frac=1-frac, random_state=random_state)
    acs_data_inj = acs_data_inj.sample(frac=frac, random_state=random_state)
    
    
    features, label, _ = ACSIncome.df_to_numpy(acs_data)
    features_inj, label_inj, _ = ACSIncome.df_to_numpy(acs_data_inj)
    if perturb is not None:
        perturb_func = perturb[0]
        perturb_param = perturb[1]
        features_inj, label_inj = perturb_func(features_inj, label_inj, perturb_param)

    # combine with injected data
    features = np.concatenate([features, features_inj])
    label = np.concatenate([label, label_inj])

#     scaler = StandardScaler()
#     scaler.fit(features)

#     X_train_scaled = scaler.transform(features)
#     X_test_scaled = scaler.transform(features_test)

#     lr = LogisticRegression(max_iter = 100000, C=0.1)
    # lr = DecisionTreeClassifier(random_state=42)
    # lr1 = RandomForestClassifier()
    lr.fit(features, label)

    preds = lr.predict(features_test)
    preds_proba = lr.predict_proba(features_test)
    
    tn, fp, fn, tp = confusion_matrix(label_test, preds, labels=[0,1]).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    tnr = tn / (tn + fp)
    if verbose:
        print(f'Accuracy: {accuracy_score(label_test, preds)}')
        print(f'Precision: {precision}')
        print(f'Recall/TPR: {recall}')
        print(f'TNR: {tnr}')
        print(f'AUC: {roc_auc_score(label_test, preds_proba[:, 1])}')
        
    return lr, preds, {'acc': accuracy_score(label_test, preds), 'tpr': recall, 'tnr': tnr}

def run_all(seed, perturb):
    lr1_ind, preds1_ind, perf1 = run_independent_n('FL', '2018', RandomForestClassifier(), n, seed)
    lr2_ind, preds2_ind, perf2 = run_independent_n('TX', '2018', RandomForestClassifier(), n, seed)

    util1_ind, util2_ind = generate_utils(preds1_ind, preds2_ind)

    # features_inj, label_inj = perturb_new_data('CA', '2018', 0.25)
    inj_data = pd.read_csv(f'data/CA_2018_1yr.csv')
    # features_inj, label_inj, _ = ACSIncome.df_to_numpy(inj_data)

    model_metadata = pd.DataFrame(columns=['frac', 'corr', 'performance1', 'performance2', 'util_diff1', 'util_diff2', 'preds1', 'preds2'])
    model_metadata.loc[len(model_metadata)] = [0, np.corrcoef(preds1_ind, preds2_ind)[0][1], perf1, perf2, np.nan, np.nan, preds1_ind, preds2_ind]

    for frac in tqdm(np.linspace(0, 1, 11)[1:]):
        lr1_corr, preds1_corr, perf1 = run_combined_n('FL', '2018', RandomForestClassifier(),
                                             inj_data, n, seed, (perturb_new_data1, perturb), frac=frac)
        lr2_corr, preds2_corr, perf2 = run_combined_n('TX', '2018', RandomForestClassifier(),
                                             inj_data, n, seed, (perturb_new_data1, perturb), frac=frac)
        corr = np.corrcoef(preds1_corr, preds2_corr)[0][1]

        util1_corr, util2_corr = generate_utils(preds1_corr, preds2_corr)

        model_metadata.loc[len(model_metadata)] = [frac, corr, perf1, perf2, util1_corr-util1_ind, util2_corr-util2_ind, preds1_corr, preds2_corr]

    fn = f'empirical/results/n_smoothinterpolation_rf_rf_perturb_{perturb}_seed_{seed}'
    model_metadata.to_pickle(f'{fn}_metadata.pkl')

seeds = [100, 20, 24, 99, 1007, 40378, 1, 3099]
for seed in seeds:
    run_all(seed, 0.25)
### END TRAINING