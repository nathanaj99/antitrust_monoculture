from folktables import ACSDataSource, ACSEmployment, ACSIncome
import pandas as pd
import numpy as np
import scipy
from sympy import *
import os
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

H, L, sigma = symbols('H L sigma')
year = '2018'

# training
state = 'CA'
data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=[state], download=True)
# acs_data = pd.read_csv(f'../empirical/data/{state}_{year}_1yr.csv')
features, label, group = ACSIncome.df_to_numpy(acs_data)

X_train, features_test, y_train, label_test = train_test_split(
    features, label, test_size=0.3, random_state=42)


def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0,1]).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    tnr = tn / (tn + fp)
    auc = roc_auc_score(y_test, preds_proba)
    accuracy = accuracy_score(y_test, preds)
    
    return preds, model, accuracy, precision, recall, tnr, auc

def run_seed(random_state):
    results_buffer = pd.DataFrame(columns=['firm', 'random_state', 'model_meta', 'preds', 'model', 'accuracy', 'precision',
                               'recall', 'tnr', 'auc'])
    
    X_train1, X_train2, y_train1, y_train2 = train_test_split(
            X_train, y_train, test_size=0.5, random_state=random_state)
    
    # SET models
    lr = LogisticRegression(solver='saga', penalty='l1')
    rf = RandomForestClassifier(n_estimators=9, min_samples_leaf=7, class_weight={0:1.2, 1:1})


    # FIRM 1
    scaler = StandardScaler()
    scaler.fit(X_train1)

    X_train_scaled = scaler.transform(X_train1)
    X_test_scaled = scaler.transform(features_test)

    l = run_model(lr, X_train_scaled, y_train1, X_test_scaled, label_test)
    results_buffer.loc[len(results_buffer)] = [1, random_state, 'lr'] + list(l)

    l = run_model(rf, X_train_scaled, y_train1, X_test_scaled, label_test)
    results_buffer.loc[len(results_buffer)] = [1, random_state, 'rf'] + list(l)


    # FIRM 2
    scaler = StandardScaler()
    scaler.fit(X_train2)

    X_train_scaled = scaler.transform(X_train2)
    X_test_scaled = scaler.transform(features_test)

    l = run_model(lr, X_train_scaled, y_train2, X_test_scaled, label_test)
    results_buffer.loc[len(results_buffer)] = [2, random_state, 'lr'] + list(l)

    l = run_model(rf, X_train_scaled, y_train2, X_test_scaled, label_test)
    results_buffer.loc[len(results_buffer)] = [2, random_state, 'rf'] + list(l)
    
    return results_buffer

def generate_utils(preds1, preds2):
    n = len(preds1)
    res = pd.DataFrame({'preds1': preds1, 'preds2': preds2, 'y': label_test})
    # follow algorithm
    hh_h = len(res[(res['preds1']) & (res['preds2']) & (res['y'])])
    hl_h = len(res[(res['preds1']) & ~(res['preds2']) & (res['y'])])
    lh_h = len(res[~(res['preds1']) & (res['preds2']) & (res['y'])])
    ll_h = len(res[~(res['preds1']) & ~(res['preds2']) & (res['y'])])
    
    hh_l = len(res[(res['preds1']) & (res['preds2']) & ~(res['y'])])
    hl_l = len(res[(res['preds1']) & ~(res['preds2']) & ~(res['y'])])
    lh_l = len(res[~(res['preds1']) & (res['preds2']) & ~(res['y'])])
    ll_l = len(res[~(res['preds1']) & ~(res['preds2']) & ~(res['y'])])
        
    util1_ff = (H/2*hh_h) + (sigma*H*hl_h) + ((1-sigma)*L*lh_h) + (L/2*ll_h) + \
                (0) + (0) + (L*lh_l) + (L/2*ll_l)
    util2_ff = (H/2*hh_h) + ((1-sigma)*L*hl_h) + (sigma*H*lh_h) + (L/2*ll_h) + \
                (0) + (0) + (L*hl_l) + (L/2*ll_l)
    
    return util1_ff, util2_ff


seeds = [100, 18456, 2, 99, 3, 910366, 10003, 204, 55, 3395, 79, 33345, 8, 524, 851]
# seed2 = [55, 18456, 3395, 79, 33345, 8, 204]

results = pd.DataFrame(columns=['firm', 'random_state', 'model_meta', 'preds', 'model', 'accuracy', 'precision',
                               'recall', 'tnr', 'auc'])

for s in tqdm(seeds):
    results = pd.concat([results, run_seed(s)], ignore_index=True)

melted_results = pd.melt(results, id_vars=['firm', 'random_state', 'model_meta', 'preds', 'model'], 
                         value_vars=['accuracy', 'precision', 'recall', 'tnr', 'auc'], 
                     var_name='metric', value_name='value')

results1 = pd.DataFrame(columns=['run', 'firm1_model', 'firm2_model', 'firm1_util', 'firm2_util', 'corr'])

step_size = 4
for idx, start in enumerate(range(0, len(results), step_size)):
    # Slice the DataFrame for the current chunk of four rows
    chunk = results.iloc[start:start + step_size]
    # Perform operations on the chunk
    
    firm1_lr = chunk[(chunk['firm'] == 1) & (chunk['model_meta'] == 'lr')].iloc[0]['preds']
    firm1_rf = chunk[(chunk['firm'] == 1) & (chunk['model_meta'] == 'rf')].iloc[0]['preds']
    firm2_lr = chunk[(chunk['firm'] == 2) & (chunk['model_meta'] == 'lr')].iloc[0]['preds']
    firm2_rf = chunk[(chunk['firm'] == 2) & (chunk['model_meta'] == 'rf')].iloc[0]['preds']
    
    for m1, m2, preds1, preds2 in [('lr', 'lr', firm1_lr, firm2_lr), ('lr', 'rf', firm1_lr, firm2_rf),
                                    ('rf', 'lr', firm1_rf, firm2_lr), ('rf', 'rf', firm1_rf, firm2_rf)]:
        util1, util2 = generate_utils(preds1, preds2)
        results1.loc[len(results1)] = [idx, m1, m2, util1, util2, np.corrcoef(preds1, preds2)[0][1]]

corr = pd.melt(results1[['run', 'firm1_model', 'firm2_model', 'corr']], id_vars=['run', 'firm1_model', 'firm2_model'], 
                         value_vars=['corr'], 
                     var_name='metric', value_name='value')
corr['models'] = corr['firm1_model'] + '-' + corr['firm2_model']
corr = corr[corr['models'].str.contains('rf-rf|lr-lr')]

corr['models'] = corr['firm1_model'] + '-' + corr['firm2_model']
corr = corr[corr['models'].str.contains('rf-rf|lr-lr')]

melted_results['model_meta'] = melted_results['model_meta'].map({'lr': 'Logistic Regression (LR)', 'rf': 'Random Forest (RF)'})
melted_results['metric'] = melted_results['metric'].map({'accuracy': 'Acc.', 'precision': 'Precision', 'recall': 'Recall', 'tnr': 'TNR', 'auc': 'AUC'})
corr['models'] = corr['models'].map({'lr-lr': 'LR-LR', 'rf-rf': 'RF-RF'})
corr['metric'] = corr['metric'].map({'corr': 'Correlation'})

# calculate difference in LR+LR - RF+RF
util_diff = pd.DataFrame(columns=['run', 'H/L', 'sigma', 'util_diff1', 'util_diff2'])
for idx, start in enumerate(range(0, len(results1), step_size)):
    # Slice the DataFrame for the current chunk of four rows
    chunk = results1.iloc[start:start + step_size]
    
    lrlr = chunk[(chunk['firm1_model'] == 'lr') & (chunk['firm2_model'] == 'lr')].iloc[0]
    rfrf = chunk[(chunk['firm1_model'] == 'rf') & (chunk['firm2_model'] == 'rf')].iloc[0]
    
    # Perform operations on the chunk
    for prop in [2, 4, 6, 8, 10, 12, 14]:
        for sig in [0.1, 0.25, 0.4]:
            util_diff.loc[(len(util_diff))] = [idx, prop, sig, 
                                (lrlr['firm1_util']-rfrf['firm1_util']).subs({'H': prop, 'L': 1, 'sigma': sig}),
                                (lrlr['firm2_util']-rfrf['firm2_util']).subs({'H': prop, 'L': 1, 'sigma': sig})]
            
util_diff['util_diff1'] = util_diff['util_diff1'].astype('float')/len(label_test)
util_diff['util_diff2'] = util_diff['util_diff2'].astype('float')/len(label_test)


### VISUALIZATION
plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(17, 4), gridspec_kw={'width_ratios': [3, 1, 1.3, 3]})

sns.barplot(data=melted_results[melted_results['firm'] == 1], x='metric', hue='model_meta', y='value', ax=ax[0])
# sns.barplot(data=melted_results[melted_results['firm'] == 2], x='metric', hue='model_meta', y='value', ax=ax[1])
sns.barplot(data=corr, x='metric', y='value', hue='models', palette='Set2', ax=ax[1])
sns.lineplot(data=util_diff, x='H/L', y='util_diff1', hue='sigma', ax=ax[3])

for i in range(2):
    ax[i].set_ylim([0.6, 1])
    ax[i].grid(True, axis='y')
    ax[i].set_xlabel('')
    if i == 1:
        ax[i].set_ylabel('')
        ax[i].set_yticklabels([])

# ax[0].set_title('Firm 1')
# ax[1].set_title('Firms 1-2')
ax[2].axis('off')

legend1 = ax[0].legend(title='Model')
legend2 = ax[1].legend(title='Models', loc='center left', bbox_to_anchor=(0.4, 0.8))

ax[3].axhline(y=0, linestyle='--')
ax[3].grid(True, axis='x')
ax[3].set_xlabel('Ratio of High to Low Price ($H/L$)')

# ax[3].set_title('Firm 1')
ax[3].set_ylabel('Util (LR-LR) - Util (RF-RF)\n(Normalized)')

legend3 = ax[3].legend(title=r'$\sigma$')

ax[0].text(-0.2, 1.15, '(a)', transform=ax[0].transAxes, fontsize=25, va='top', weight='bold')
ax[3].text(-0.3, 1.15, '(b)', transform=ax[3].transAxes, fontsize=25, va='top', weight='bold'
           )
ax[3].text(0.4, 0.92, '\u2191 $\sigma$ = \u2193 Price Sensitivity', transform=ax[3].transAxes, fontsize=16, va='top',
           bbox=dict(facecolor='none', edgecolor='lightgrey', boxstyle='round,pad=0.5'))
fig.text(0.35, -0.02, 'Metric', ha='center', va='center')


plt.subplots_adjust(right=0.95, wspace=0.05, hspace=0.1)
plt.savefig('../figs/empirical_models_main.pdf', bbox_inches='tight')
plt.show()



###### FIRST-STAGE GAME
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from matplotlib.lines import Line2D

res = pd.DataFrame()

# calculate difference in LR+LR - RF+RF
util_diff = pd.DataFrame(columns=['run', 'H/L', 'sigma', 'util_diff1', 'util_diff2'])
for idx, start in enumerate(range(0, len(results), step_size)):
    # Slice the DataFrame for the current chunk of four rows
    chunk = results.iloc[start:start + step_size]

    n = None
    
    results_buffer = pd.DataFrame(columns=['model1', 'model2', 'util1_ff', 'util2_ff', 'util1_hf', 'util2_hf', 
                                'util1_fh', 'util2_fh', 'util1_lf', 'util2_lf', 'util1_fl', 'util2_fl'])
    possible_frac = ['lr', 'rf']

    for frac2 in tqdm(possible_frac):
        preds2 = chunk[(chunk['firm'] == 2) & (chunk['model_meta'] == frac2)].iloc[0]['preds']
        n = len(preds2)
        preds2_h = [True]*n
        preds2_l = [False]*n

        # go through player 1 all frac, then pick the best gamma utility
        for frac1 in possible_frac:
            preds1 = chunk[(chunk['firm'] == 1) & (chunk['model_meta'] == frac1)].iloc[0]['preds']
            preds1_h = [True]*n
            preds1_l = [False]*n

            util1_ff, util2_ff = generate_utils(preds1, preds2)
            util1_hf, util2_hf = generate_utils(preds1_h, preds2)
            util1_fh, util2_fh = generate_utils(preds1, preds2_h)
            util1_lf, util2_lf = generate_utils(preds1_l, preds2)
            util1_fl, util2_fl = generate_utils(preds1, preds2_l)
            util1_hh, util2_hh = generate_utils(preds1, preds2)
            util1_ll, util2_ll = generate_utils(preds1, preds2)

            results_buffer.loc[len(results_buffer)] = [frac1, frac2, util1_ff, util2_ff, util1_hf, util2_hf,
                                        util1_fh, util2_fh, util1_lf, util2_lf, util1_fl, util2_fl]
    
    res = pd.concat([res, results_buffer], ignore_index=True)

agg = res.groupby(['model1', 'model2']).agg('sum').reset_index()
for i in ['util1_ff', 'util2_ff', 'util1_hf', 'util2_hf',
       'util1_fh', 'util2_fh', 'util1_lf', 'util2_lf', 'util1_fl', 'util2_fl']:
    agg[i] /= len(res)/4

def eq_analysis_list(l):
    plt.rcParams.update({'font.size': 26})
    fig, ax = plt.subplots(nrows=1, ncols=len(l), sharey=True, figsize=(20, 5))
    
    for j, fixed in enumerate(l):
        util_lrlr1 = None
        util_lrlr2 = None
        util_rfrf1 = None
        util_rfrf2 = None

        # 2nd stage equilibrium: Both firms need to prefer following algorithm over alternatives
        eq_in_2nd = pd.DataFrame(columns=list(agg.columns))
        not_eq = []
        for idx, row in agg.iterrows():
        # firm 1
            ff = row['util1_ff'].subs(fixed)
            lf = row['util1_lf'].subs(fixed)
            hf = row['util1_hf'].subs(fixed)

            ff2 = row['util2_ff'].subs(fixed)
            fl2 = row['util2_fl'].subs(fixed)
            fh2 = row['util2_fh'].subs(fixed)
            print(row['model1'], row['model2'], ff, ff2)

            if row['model1'] == 'lr' and row['model2'] == 'lr':
                util_lrlr1 = ff
                util_lrlr2 = ff2
            elif row['model1'] == 'rf' and row['model2'] == 'rf':
                util_rfrf1 = ff
                util_rfrf2 = ff2

            if ff >= lf and ff >= hf and ff2 >= fl2 and ff2 >= fh2:
                eq_in_2nd.loc[len(eq_in_2nd)] = row.tolist()
            else:
                not_eq.append((row['model1'], row['model2']))
                
        # now go through and record best response for 1st-stage game
        br1 = set()
        br2 = set()

        # go through firm 1 first. fix firm 2 gamma, and pick the largest firm 1 utility
        for frac2 in possible_frac:
            subset = eq_in_2nd[eq_in_2nd['model2'] == frac2]
            save_res = {}
            for idx, row in subset.iterrows():
                save_res[row['model1']] = row['util1_ff'].subs(fixed)
            if not save_res:
                pass
            else:
                highest = max(save_res, key=save_res.get)
                br1.add((highest, frac2))

        for frac1 in possible_frac:
            subset = eq_in_2nd[eq_in_2nd['model1'] == frac1]
            save_res = {}
            for idx, row in subset.iterrows():
                save_res[row['model2']] = row['util2_ff'].subs(fixed)
            if not save_res:
                pass
            else:
                highest = max(save_res, key=save_res.get)
                br2.add((frac1, highest))

        # remap to positions -- relevant for plotting image
        mapping = {'lr': 0, 'rf': 1}
        br1_pos = []
        for i in list(br1):
            br1_pos.append((mapping[i[0]], mapping[i[1]]))

        br2_pos = []
        for i in list(br2):
            br2_pos.append((mapping[i[0]], mapping[i[1]]))

        not_eq_pos = []
        for i in list(not_eq):
            not_eq_pos.append((mapping[i[0]], mapping[i[1]]))

        n = 2
        matrix = np.ones((n, n))
    
        # Plot the matrix
        cax = ax[j].imshow(matrix, cmap=ListedColormap(['white', 'lightgray']), interpolation='nearest')
    #     plt.colorbar(cax)

        # Highlight the specified cells
        for (row, col) in br1_pos:
            if br1_pos == [(0, 0), (1, 1)] and br2_pos == [(0, 0), (1, 1)] and row == 0 and col == 0:
                if util_lrlr1 > util_rfrf1 and util_lrlr2 > util_rfrf2:
                    rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=4, edgecolor='red', facecolor='#f2f7c6', alpha=0.5)
                elif util_lrlr1 < util_rfrf1 and util_lrlr2 < util_rfrf2 and row == 1 and col == 1:
                    rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=4, edgecolor='red', facecolor='#f2f7c6', alpha=0.5)
                else:
                    rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=4, edgecolor='red', facecolor='none', alpha=0.5)
            else:
                rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=4, edgecolor='red', facecolor='none', alpha=0.5)
            diag_line = Line2D([col-0.5, col+0.5], [row-0.5, row+0.5], color='red', linewidth=2)
            ax[j].add_patch(rect)
            ax[j].add_line(diag_line)
            # ax[j].add_patch(rect)

        for (row, col) in br2_pos:
            # Create a rectangle around the cell
            if br1_pos == [(0, 0), (1, 1)] and br2_pos == [(0, 0), (1, 1)]:
                if util_lrlr1 > util_rfrf1 and util_lrlr2 > util_rfrf2 and row == 0 and col == 0:
                    rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=4, edgecolor='blue', facecolor='#f2f7c6', alpha=0.5)
                elif util_lrlr1 < util_rfrf1 and util_lrlr2 < util_rfrf2 and row == 1 and col == 1:
                    rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=4, edgecolor='blue', facecolor='#f2f7c6', alpha=0.5)
                else:
                    rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=4, edgecolor='blue', facecolor='none', alpha=0.5)
            else:
                rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=4, edgecolor='blue', facecolor='none', alpha=0.5)
            diag_line = Line2D([col-0.5, col+0.5], [row+0.5, row-0.5], color='blue', linewidth=2)
            ax[j].add_patch(rect)
            ax[j].add_line(diag_line)

        for (row, col) in not_eq_pos:
            # Create a rectangle around the cell
            rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=4, edgecolor='none', facecolor='grey', alpha=0.5)
            ax[j].add_patch(rect)


        # Set the ticks and labels
        ax[j].set_xticks(np.arange(n))
        ax[j].set_yticks(np.arange(n))

        tic = ['LR', 'RF']

        ax[j].set_xticklabels(tic)
        ax[j].set_yticklabels(tic)

         # Draw grid lines
        ax[j].grid(which='both', color='black', linestyle='-', linewidth=1)
        ax[j].set_xticks(np.arange(0.5, n, 1), minor=True)
        ax[j].set_yticks(np.arange(0.5, n, 1), minor=True)
        ax[j].grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax[j].grid(which='major', color='none')

        # Hide major tick marks
        ax[j].tick_params(axis='both', which='major', length=0)

        ax[j].tick_params(axis='x', labeltop=True, labelbottom=False)
        ax[j].set_title(r'$H/L={}, \sigma={}$'.format(fixed['H'], 
                                                      fixed['sigma']), size=24)
        
        if j == 0:
            ax[j].set_ylabel(r'Firm 1 Model')
    fig.text(0.5, 0.15, r'Firm 2 Model', va='center', rotation='horizontal')
    fig.text(0.3, 1.03, '\u2191 $\sigma$ = \u2193 Price Sensitivity', fontsize=20, va='top',
           bbox=dict(facecolor='none', edgecolor='lightgrey', boxstyle='round'))
    
    fig.text(0.5, 1.03, '$H/L$ = Ratio of High to Low Price', fontsize=20, va='top',
           bbox=dict(facecolor='none', edgecolor='lightgrey', boxstyle='round'))

        # Display the plot
    plt.subplots_adjust(right=0.97, wspace=0.3, hspace=0.15)
    plt.savefig('../figs/empirical_models_eq_selection.pdf', bbox_inches='tight')
    plt.show()

l = [{'sigma': 0.3, 'H': 1.5, 'L': 1},
     {'sigma': 0.3, 'H': 3, 'L': 1},
     {'sigma': 0.1, 'H': 7, 'L': 1},
    {'sigma': 0.2, 'H': 10, 'L': 1},
    {'sigma': 0.3, 'H': 10, 'L': 1}]
eq_analysis_list(l)