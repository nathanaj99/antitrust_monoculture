import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from folktables import ACSDataSource, ACSEmployment, ACSIncome
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm

### ANALYSIS: PREFERENCES
perturb = 0.25
seeds = [100, 20, 24, 99, 1007, 40378, 1, 3099]

model_metadata = pd.DataFrame()
for seed in seeds:
    fn = f'../empirical/results/n_smoothinterpolation_rf_rf_perturb_{perturb}_seed_{seed}'
    buffer = pd.read_pickle(f'{fn}_metadata.pkl')
    model_metadata = pd.concat([model_metadata, buffer], ignore_index=True)
    
perf = pd.DataFrame(columns=['frac', 'corr', 'acc1', 'tpr1', 'tnr1', 'acc2', 'tpr2', 'tnr2'])
for idx, row in model_metadata.iterrows():
    perf1 = row['performance1']
    perf2 = row['performance2']
    perf.loc[len(perf)] = [row['frac'], row['corr'], perf1['acc'], perf1['tpr'], perf1['tnr'],
                          perf2['acc'], perf2['tpr'], perf2['tnr']]
    
n = len(model_metadata.iloc[0]['preds1'])

plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(15, 2))
ax[1].sharey(ax[2])
sns.lineplot(data=perf, x='frac', y='corr', ax=ax[0])
ax[0].set_ylabel('Correlation')
sns.lineplot(data=perf, x='frac', y='acc1', ax=ax[1])
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Firm 1')
sns.lineplot(data=perf, x='frac', y='acc2', ax=ax[2])
ax[2].set_ylabel('Accuracy')
ax[2].set_title('Firm 2')

for i in range(3):
    if i == 1:
        ax[i].set_xlabel(r'$\gamma$')
    else:
        ax[i].set_xlabel('')

    ax[i].grid(True, axis='both')

plt.subplots_adjust(right=0.95, wspace=0.4, hspace=0.1)
fig.text(0.08, 1.06, '(a)',  fontsize=24, fontweight='bold', va='top', ha='right')

plt.savefig('../figs/empirical_metadata.pdf', bbox_inches='tight')
plt.show()

df = pd.DataFrame(columns=['frac', 'H/L', 'sigma', 'util_diff1', 'util_diff2'])
for idx, row in model_metadata.iterrows():
    for prop in [2, 3, 5]:
        for sig in [0.2, 0.3, 0.4]:
            if not pd.isna(row['util_diff1']):
                calc1 = row['util_diff1'].subs({'H': prop, 'L': 1, 'sigma': sig})/n
                calc2 = row['util_diff2'].subs({'H': prop, 'L': 1, 'sigma': sig})/n
                df.loc[len(df)] = [row['frac'], prop, sig, calc1, calc2]
                              
df['util_diff1'] = df['util_diff1'].astype('float')
df['util_diff2'] = df['util_diff2'].astype('float')

# plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(15, 5))

for idx, prop in enumerate([2, 3, 5]):
    subset = df[df['H/L'] == prop]
    sns.lineplot(data=subset, x='frac', y='util_diff1', hue='sigma', ax=ax[0, idx])
    sns.lineplot(data=subset, x='frac', y='util_diff2', hue='sigma', ax=ax[1, idx])
    for i in range(2):
        ax[i, idx].axhline(y=0, linestyle='--')
        
    leg = ax[0, idx].get_legend()
    leg.remove()
    if idx != 2:
        leg = ax[1, idx].get_legend()
        leg.remove()
    if idx == 2:
        ax[1, idx].legend(bbox_to_anchor=(1.5, 1.05), title=r'$\sigma$')
    if idx == 0:
        ax[0, idx].set_ylabel('Firm 1')
        ax[1, idx].set_ylabel('Firm 2')
    else:
        ax[0, idx].set_ylabel('')
        ax[1, idx].set_ylabel('')
        
    for i in range(2):
        if idx == 1 and i == 1:
            ax[i, idx].set_xlabel(r'$\gamma$')
        else:
            ax[i, idx].set_xlabel('')

        for j in range(3):
            ax[i, j].grid(True, axis='both')
            
    ax[0, idx].set_title(f'H/L = {prop}')
        
# Add a shared y-axis label between the first column subplots
fig.text(0.015, 0.5, r'$U(\gamma) - U(\gamma = 0)$ (Normalized)', va='center', rotation='vertical')

fig.text(0.05, 1.0, '(b)',  fontsize=24, fontweight='bold', va='top', ha='right')

#     if idx == 1:
        

plt.subplots_adjust(right=0.95, wspace=0.3, hspace=0.1)
plt.savefig('../figs/empirical_util_diff.pdf', bbox_inches='tight')
plt.show()


### ANALYSIS: EQUILIBRIUM SELECTION
def create_matrix_with_highlights(n, highlighted_cells1, highlighted_cells2, invalid_cells):
    """
    Creates an n by n matrix and highlights specific cells.
    
    Parameters:
    n (int): The size of the matrix.
    highlighted_cells (list of tuples): A list of tuples where each tuple represents the coordinates (row, col)
                                        of a cell to be highlighted.
    """
    matrix = np.ones((n, n))
    
    # Plot the matrix
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap=ListedColormap(['white', 'lightgray']), interpolation='nearest')
#     plt.colorbar(cax)
    
    # Highlight the specified cells
    for (row, col) in highlighted_cells1:
        # Create a rectangle around the cell
        rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=3, edgecolor='red', facecolor='none', alpha=0.5)
        ax.add_patch(rect)
        
    for (row, col) in highlighted_cells2:
        # Create a rectangle around the cell
        rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=3, edgecolor='blue', facecolor='none', alpha=0.5)
        ax.add_patch(rect)
        
    for (row, col) in invalid_cells:
        # Create a rectangle around the cell
        rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=3, edgecolor='none', facecolor='grey', alpha=0.5)
        ax.add_patch(rect)
    
    
    # Set the ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
#     ax.set_xticklabels(np.arange(1, n+1))
#     ax.set_yticklabels(np.arange(1, n+1))
    
    tic = np.around(np.linspace(0, 1, n), 2)
    ax.set_xticklabels(tic)
    ax.set_yticklabels(tic)
    
     # Draw grid lines
    ax.grid(which='both', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(0.5, n, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.grid(which='major', color='none')
    
    # Hide major tick marks
    ax.tick_params(axis='both', which='major', length=0)
    
    ax.tick_params(axis='x', labeltop=True, labelbottom=False)
    
    # Display the plot
    plt.show()

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

H, L, sigma = symbols('H L sigma')
n = 200000
# test data -- this is a constant

test_state = 'IL'
year = '2018'

test_data = pd.read_csv(f'../empirical/data/{test_state}_{year}_1yr.csv')
n_test = len(test_data)
features_test, label_test, _ = ACSIncome.df_to_numpy(test_data)

perturb = 0.25
seeds = [100, 20, 24, 99, 1007, 40378, 1, 3099]

results = pd.DataFrame()
for seed in seeds:
    fn = f'../empirical/results/n_smoothinterpolation_rf_rf_perturb_{perturb}_seed_{seed}'
    model_metadata = pd.read_pickle(f'{fn}_metadata.pkl')
    model_metadata['frac'] = model_metadata['frac'].apply(lambda x: round(x, 2))
    
    n = len(model_metadata.iloc[0]['preds1'])
    
    results_buffer = pd.DataFrame(columns=['frac1', 'frac2', 'util1_ff', 'util2_ff', 'util1_hf', 'util2_hf', 
                                'util1_fh', 'util2_fh', 'util1_lf', 'util2_lf', 'util1_fl', 'util2_fl'])
    possible_frac = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for frac2 in tqdm(possible_frac):
        preds2 = model_metadata[model_metadata['frac'] == frac2].iloc[0]['preds2']
        preds2_h = [True]*n
        preds2_l = [False]*n

        # go through player 1 all frac, then pick the best gamma utility
        for frac1 in possible_frac:
            preds1 = model_metadata[model_metadata['frac'] == frac1].iloc[0]['preds1']
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
    
    results = pd.concat([results, results_buffer], ignore_index=True)

agg = results.groupby(['frac1', 'frac2']).agg('sum').reset_index()
for i in ['util1_ff', 'util2_ff', 'util1_hf', 'util2_hf',
       'util1_fh', 'util2_fh', 'util1_lf', 'util2_lf', 'util1_fl', 'util2_fl']:
    agg[i] /= len(seeds)

def eq_analysis_list(l):
    plt.rcParams.update({'font.size': 25})
    fig, ax = plt.subplots(nrows=1, ncols=len(l), sharey=True, figsize=(20, 15))
    
    for j, fixed in enumerate(l):
    
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
            if ff >= lf and ff >= hf and ff2 >= fl2 and ff2 >= fh2:
                eq_in_2nd.loc[len(eq_in_2nd)] = row.tolist()
            else:
                not_eq.append((row['frac1'], row['frac2']))

        # now go through and record best response for 1st-stage game
        br1 = set()
        br2 = set()

        # go through firm 1 first. fix firm 2 gamma, and pick the largest firm 1 utility
        for frac2 in possible_frac:
            subset = eq_in_2nd[eq_in_2nd['frac2'] == frac2]
            save_res = {}
            for idx, row in subset.iterrows():
                save_res[row['frac1']] = row['util1_ff'].subs(fixed)
            if not save_res:
                pass
            else:
                highest = max(save_res, key=save_res.get)
                br1.add((highest, frac2))

        for frac1 in possible_frac:
            subset = eq_in_2nd[eq_in_2nd['frac1'] == frac1]
            save_res = {}
            for idx, row in subset.iterrows():
                save_res[row['frac2']] = row['util2_ff'].subs(fixed)
            if not save_res:
                pass
            else:
                highest = max(save_res, key=save_res.get)
                br2.add((frac1, highest))

        # remap to positions -- relevant for plotting image
        mapping = {i/10: i for i in range(11)}
        br1_pos = []
        for i in list(br1):
            br1_pos.append((mapping[i[0]], mapping[i[1]]))

        br2_pos = []
        for i in list(br2):
            br2_pos.append((mapping[i[0]], mapping[i[1]]))

        not_eq_pos = []
        for i in list(not_eq):
            not_eq_pos.append((mapping[i[0]], mapping[i[1]]))

        n = 11
        matrix = np.ones((n, n))
    
        # Plot the matrix
        cax = ax[j].imshow(matrix, cmap=ListedColormap(['white', 'lightgray']), interpolation='nearest')
    #     plt.colorbar(cax)

        # Highlight the specified cells
        for (row, col) in br1_pos:
            # Create a rectangle around the cell
            rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=4, edgecolor='red', facecolor='none', alpha=0.5)
            diag_line = Line2D([col-0.5, col+0.5], [row-0.5, row+0.5], color='red', linewidth=2)
            ax[j].add_patch(rect)
            ax[j].add_line(diag_line)

        for (row, col) in br2_pos:
            # Create a rectangle around the cell
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
    #     ax.set_xticklabels(np.arange(1, n+1))
    #     ax.set_yticklabels(np.arange(1, n+1))

        tic = []
        for i in np.around(np.linspace(0, 1, 11), 2):
            if i == 1:
                tic.append(str(i)[0])
            else:
                tic.append(str(i)[1:])
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
            ax[j].set_ylabel(r'$\gamma$ (Firm 1)')
    fig.text(0.5, 0.32, r'$\gamma$ (Firm 2)', va='center', rotation='horizontal')

        # Display the plot
    plt.subplots_adjust(right=0.97, wspace=0.3, hspace=0.15)
    plt.savefig('../figs/empirical_eq_selection.pdf', bbox_inches='tight')
    plt.show()

l = [{'sigma': 0.2, 'H': 3, 'L': 1},
     {'sigma': 0.1, 'H': 6, 'L': 1},
    {'sigma': 0.1, 'H': 10, 'L': 1}]
eq_analysis_list(l)