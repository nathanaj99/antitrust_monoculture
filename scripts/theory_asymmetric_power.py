import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy
import matplotlib.pyplot as plt
from sympy import *
import sympy
import seaborn as sns
from tqdm import tqdm
import copy
import matplotlib
import scipy.special
import time
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, LineString, Point, MultiPoint, MultiPolygon
from shapely import unary_union
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

a_i, a_j, rho, sigma, theta, H, L, delta_H, delta_L, gamma = symbols('a_i a_j rho sigma theta H L delta_H delta_L gamma')

p11 = (a_i * a_j + rho)
p10 = (a_i * (1-a_j) - rho)
p01 = (a_j * (1-a_i) - rho)
p00 = (1 - a_i - a_j + a_i*a_j + rho)


# utility of player i when they get p_i = 1 and price high
u_h_1 = H * gamma * theta * p11 + sigma * gamma * H * theta * p10 # omitting when tau_L because their payoff is going to be 0
# utility of player i when they get p_i = 1 and price low
u_l_1 = (1-sigma+gamma * sigma) * L * theta * p11 + L*gamma * theta * p10 + L*(1-theta) * p00 + L * gamma * (1-theta) * p01

# utility of player i when they get p_i = 0 and price high
u_h_0 = H * gamma * theta * p01 + sigma * gamma * H * theta * p00

# utility of player i when they get p_i = 0 and price low
u_l_0 = (1-sigma + gamma * sigma) * L * theta * p01 + L*gamma * theta * p00 + L*(1-theta) * p10 + L * gamma * (1-theta) * p11

ex1 = u_h_1 - u_l_1
ex0 = u_h_0 - u_l_0


# only need to use this if there is any asymmetry in the market
u_h_1_j = H * (1-gamma) * theta * p11 + sigma * (1-gamma) * H * theta * p01
u_l_1_j = (1-gamma*sigma) * L * theta * p11 + L * (1-gamma) * theta * p01 + L*(1-theta) * p00 + L * (1-gamma) * (1-theta) * p10
u_h_0_j = H * (1-gamma) * theta * p10 + sigma * (1-gamma) * H * theta * p00
u_l_0_j = (1-gamma*sigma) * L * theta * p10 + L * (1-gamma) * theta * p00 + L*(1-theta) * p01 + L * (1-gamma) * (1-theta) * p11

ex1_j = u_h_1_j - u_l_1_j
ex0_j = u_h_0_j - u_l_0_j

w_h_1 = delta_H * theta * p11 + (delta_H + (1-sigma)*(H-L)) * theta * p10 + (delta_L) * (1-theta) * p01
w_l_0 = (delta_H + (1-sigma)*(H-L)) * theta * p01 + (delta_H + (H-L)) * theta * p00 +\
        delta_L*(1-theta) * p10 + delta_L * (1-theta) * p11

cmap = sns.diverging_palette(20, 255, s=100, l=35, as_cmap=True, center='light')

def get_vals(expr, x, y, x_var, y_var, vals):
    t = expr.subs(vals)
    # print(t)
    f = lambdify([x_var, y_var], t, 'numpy')
    
    res = f(x, y)
    return res

x,y = np.meshgrid(np.linspace(0, 1, 501), np.linspace(0, 1, 1001)[:-1])
prop_list = [2, 4]
theta_list = [0.5, 0.75]

acc = 0.9


# cmap_bin = ListedColormap(['white', 'blue'])

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(15, 8))
for i, s in enumerate(prop_list):
    for j, t in enumerate(theta_list):
        vals = {'rho': 0, 'a_j': acc, 'a_i': acc, 'theta': t, 'H': s, 'L': 1}
        res0 = get_vals(ex0 <= 0, x, y, gamma, sigma, vals)
        res1 = get_vals(ex1 >= 0, x, y, gamma, sigma, vals)

        res0_j = get_vals(ex0_j <= 0, x, y, gamma, sigma, vals)
        res1_j = get_vals(ex1_j >= 0, x, y, gamma, sigma, vals)
        
        vals_same = {'a_j': acc, 'a_i': acc, 'theta': t, 'H': s, 'L': 1}
        res0_same = get_vals(ex0.subs({'rho': Piecewise((a_i, a_i < a_j), (a_j, True))-a_i*a_j}) <= 0, 
                             x, y, gamma, sigma, vals_same)
        res1_same = get_vals(ex1.subs({'rho': Piecewise((a_i, a_i < a_j), (a_j, True))-a_i*a_j}) >= 0, 
                             x, y, gamma, sigma, vals_same)
        
        res0_same_j = get_vals(ex0_j.subs({'rho': Piecewise((a_i, a_i < a_j), (a_j, True))-a_i*a_j}) <= 0, 
                             x, y, gamma, sigma, vals_same)
        res1_same_j = get_vals(ex1_j.subs({'rho': Piecewise((a_i, a_i < a_j), (a_j, True))-a_i*a_j}) >= 0, 
                             x, y, gamma, sigma, vals_same)
        
        ax[i,j].imshow((res0 & res1 & res0_j & res1_j).astype(int),
                       origin="lower", cmap=ListedColormap(['none', 'lightgray']),
                  extent=(x.min(),x.max(),y.min(),y.max()), aspect='auto', label='Player i')
        ax[i,j].imshow((res0_same & res1_same & res0_same_j & res1_same_j).astype(int),
                       origin="lower", cmap=ListedColormap(['none', 'gray']),
                  extent=(x.min(),x.max(),y.min(),y.max()), aspect='auto', label='Player i')
        
        # ax[i,j].imshow((res1_same & res0_same).astype(int),
        #                origin="lower", cmap=ListedColormap(['none', 'gray']),
        #           extent=(x.min(),x.max(),y.min(),y.max()), aspect='auto', label='Same alg.')
        
        condition = res0 & res1 & res0_same & res1_same & res0_j & res1_j & res0_same_j & res1_same_j
        
        Z = get_vals(u_h_1 + u_l_0, x, y, gamma, sigma, vals)
        Z_correlated = get_vals((u_h_1 + u_l_0).subs({'rho': Piecewise((a_i, a_i < a_j), (a_j, True))-a_i*a_j}),
                                  x, y, gamma, sigma, vals_same)
        
        Z_j = get_vals(u_h_1_j + u_l_0_j, x, y, gamma, sigma, vals)
        Z_correlated_j = get_vals((u_h_1_j + u_l_0_j).subs({'rho': Piecewise((a_i, a_i < a_j), (a_j, True))-a_i*a_j}),
                                  x, y, gamma, sigma, vals_same)

        Z_masked = np.ma.array(Z, mask=~condition)
        Z_corr_masked = np.ma.array(Z_correlated, mask=~condition)

        Z_j_masked = np.ma.array(Z_j, mask=~condition)
        Z_corr_j_masked = np.ma.array(Z_correlated_j, mask=~condition)

        diff = Z_corr_masked - Z_masked
        diff_j = Z_corr_j_masked - Z_j_masked

        im = ax[i, j].imshow(((diff > 0) & (diff_j > 0)).astype(int), origin="lower", cmap=ListedColormap(['#ffb3b3', '#b3b3ff']),
                             extent=(x.min(),x.max(),y.min(),y.max()), aspect='auto', vmin=-0.1, vmax=0.1)
        # im_j = ax[i, j].imshow(diff_j, origin="lower", cmap=cmap,
        #                      extent=(x.min(),x.max(),y.min(),y.max()), aspect='auto', vmin=-0.1, vmax=0.1)
        
        ax[i, j].axvline(x=0.5, linestyle='--', color='black', alpha=0.5)
        if i == 0:
            ax[i, j].set_title(r'$\theta$ = {}'.format(t))
        
        if i == 1:
            ax[i, j].set_xlabel(r'$\gamma$')
        if j == 0:
            ax[i, j].set_ylabel(r'$\sigma$')
        


        # if i == 1 and j == 1:
#             cbar = fig.colorbar(im, ticks=[0, 1], ax=ax, label='Region')
#             cbar.clim(-0.5, 1.5)

            # cbar_ax = fig.add_axes([1.02, 0.1, 0.02, 0.5])
            # fig.colorbar(im, cax=cbar_ax, orientation='vertical')

# Define custom patches for the legend
custom_patches = [
    Patch(color='lightgray', label='Independent'),
    Patch(color='gray', label='Same alg.'),
]

# Add the legend to the figure
fig.legend(handles=custom_patches, loc='center left', bbox_to_anchor=(1, 0.75))


custom_patches1 = [
    Patch(color='#ffb3b3', label='One or both prefer\nindependence'),
    Patch(color='#b3b3ff', label='Prefer correlation')
]

# Add the legend to the figure
fig.legend(handles=custom_patches1, loc='center left', bbox_to_anchor=(1, 0.6))

# Adjust the layout to make space for the legend
plt.subplots_adjust(right=0.98, wspace=0.1, hspace=0.15)


rows = [r'$H/L$ = {}'.format(row) for row in prop_list]

pad = 5 # in points

for aa, row in zip(ax[:,0], rows):
    aa.annotate(row, xy=(0, 0.5), xytext=(-aa.yaxis.labelpad - pad, 0),
                xycoords=aa.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
    
fig.text(0.1, 1.0, '\u2191 $\sigma$ = \u2193 Price Sensitivity', fontsize=22, va='top',
           bbox=dict(facecolor='none', edgecolor='lightgrey', boxstyle='round'))
    
fig.text(0.38, 1.0, '$H/L$ = Ratio of High to Low Price', fontsize=22, va='top',
        bbox=dict(facecolor='none', edgecolor='lightgrey', boxstyle='round'))

fig.text(0.73, 1.0, r'$\theta$ = % Population Willing to Pay $H$', fontsize=22, va='top',
        bbox=dict(facecolor='none', edgecolor='lightgrey', boxstyle='round'))
    
# fig.tight_layout()

plt.savefig('../figs/brand_loyalty.pdf', bbox_inches='tight')
plt.show()