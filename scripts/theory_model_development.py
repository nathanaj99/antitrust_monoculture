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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Patch


a_i, a_j, rho, sigma, theta, H, L, delta_H, delta_L = symbols('a_i a_j rho sigma theta H L delta_H delta_L')
a_c, epsilon = symbols('a_c epsilon')

bias_term = Piecewise((a_i, a_i < a_j), (a_j, True))-a_i*a_j

p11 = (a_i * a_j + rho * bias_term)
p10 = (a_i * (1-a_j) - rho * bias_term)
p01 = (a_j * (1-a_i) - rho * bias_term)
p00 = (1 - a_i - a_j + a_i*a_j + rho * bias_term)

u_h_1 = H/2 * theta * p11 + sigma*H* theta * p10
u_l_1 = (1-sigma) * L * theta * p11 + L/2 * theta * p10 + L*(1-theta) * p00 + L/2 * (1-theta) * p01
u_h_0 = H/2 * theta * p01 + sigma * H * theta * p00
u_l_0 = (1-sigma) * L * theta * p01 + L/2 * theta * p00 + L*(1-theta) * p10 + L/2 * (1-theta) * p11

ex1 = u_h_1 - u_l_1
ex0 = u_h_0 - u_l_0

# only need to use this if a_i \neq a_j
u_h_1_j = H/2 * theta * p11 + sigma*H* theta * p01
u_l_1_j = (1-sigma) * L * theta * p11 + L/2 * theta * p01 + L*(1-theta) * p00 + L/2 * (1-theta) * p10
u_h_0_j = H/2 * theta * p10 + sigma * H * theta * p00
u_l_0_j = (1-sigma) * L * theta * p10 + L/2 * theta * p00 + L*(1-theta) * p01 + L/2 * (1-theta) * p11

ex1_j = u_h_1_j - u_l_1_j
ex0_j = u_h_0_j - u_l_0_j

# w_h_1 = delta_H * theta * p11 + (delta_H + (1-sigma)*(H-L)) * theta * p10 + (delta_L) * (1-theta) * p01
# w_l_0 = (delta_H + (1-sigma)*(H-L)) * theta * p01 + (delta_H + (H-L)) * theta * p00 +\
#         delta_L*(1-theta) * p10 + delta_L * (1-theta) * p11

def mult(l):
    x, y = np.meshgrid(np.linspace(0.5, 1, 1001)[:-1], np.linspace(0.5, 1, 1001)[:-1])
    plt.rcParams.update({'font.size': 25})
    cmap = sns.diverging_palette(20, 255, s=100, l=35, as_cmap=True, center='light')
    n = len(l)
    fig, ax = plt.subplots(nrows=1, ncols=n, sharey=True, figsize=(15, 3))
    for i, item in enumerate(l):
        fixed = item['fixed']
        rho_corr = item['rho_corr']
    
        eq = ((u_h_1 + u_l_0).subs({'a_j': a_i}) - \
              (u_h_1 + u_l_0).subs({'a_i': a_c, 'a_j': a_i})).subs(fixed).subs({'rho': 0})
        f_ind = lambdify([a_c, a_i], eq, 'numpy')

        # correaltion condition
        eq_corr = ((u_h_1 + u_l_0).subs({'a_i': a_c, 'a_j': a_c, 'rho': rho_corr}) - \
                   (u_h_1 + u_l_0).subs({'a_j': a_c, 'rho': 0})).subs(fixed)
        f_corr = lambdify([a_c, a_i], eq_corr, 'numpy')

        res_corr = f_corr(x, y)
        res_ind = f_ind(x, y)
        both_in_eq = (res_ind >= 0) & (res_corr >= 0)

        # mask regions where 2nd-stage game is not in equilibrium
        eq_ind1 = ex1.subs({'a_j': a_i}).subs(fixed).subs({'rho': 0})
        eq_ind0 = ex0.subs({'a_j': a_i}).subs(fixed).subs({'rho': 0})

        eq_corr1 = ex1.subs({'a_i': a_c, 'a_j': a_c}).subs(fixed).subs({'rho': rho_corr})
        eq_corr0 = ex0.subs({'a_i': a_c, 'a_j': a_c}).subs(fixed).subs({'rho': rho_corr})

        f_ind1 = lambdify([a_c, a_i], eq_ind1, 'numpy')
        f_ind0 = lambdify([a_c, a_i], eq_ind0, 'numpy')
        f_corr1 = lambdify([a_c, a_i], eq_corr1, 'numpy')
        f_corr0 = lambdify([a_c, a_i], eq_corr0, 'numpy')

        res1_ind = f_ind1(x, y)
        res0_ind = f_ind0(x, y)

        res1_corr = f_corr1(x, y)
        res0_corr = f_corr0(x, y)

        second_stage_condition = (res1_ind >= 0) & (res0_ind <= 0) & (res1_corr >= 0) & (res0_corr <= 0)

        Z_corr = (u_h_1 + u_l_0).subs({'a_i': a_c, 'a_j': a_c, 'rho': rho_corr}).subs(fixed)
        Z_ind = (u_h_1 + u_l_0).subs({'a_j': a_i, 'rho': 0}).subs(fixed)
        f_util = lambdify([a_c, a_i], Z_corr-Z_ind, 'numpy')
        Z = f_util(x, y)

        # mask based on first stage and 2nd stage condition
        Z_masked = np.ma.array(Z, mask=~second_stage_condition)
        Z_masked1 = np.ma.array(Z_masked, mask=~both_in_eq)

        im = ax[i].imshow(Z_masked1, origin="lower", cmap=cmap,
                          extent=(x.min(),x.max(),y.min(),y.max()), aspect='auto', vmax=0.1, vmin=-0.1)
        
        if i == 0:
            cbar_ax = fig.add_axes([0.98, 0.1, 0.015, 0.9])
            fig.colorbar(im, cax=cbar_ax, orientation='vertical')
            ax[i].set_ylabel('Perf. of independent\nmodel ($a_i$)')
            
            
        ax[i].plot([0, 1], [0, 1], transform=ax[i].transAxes, linestyle='--')
        if i == 1:
            ax[i].set_xlabel(r'Performance of correlated model ($a_c$)')
        else:
            ax[i].set_xlabel('')
        ax[i].set_title(r'$\sigma={}, \rho_c={}$'.format(fixed['sigma'], rho_corr), size=24)

        ax[i].grid(True, axis='both')

    fig.text(0.27, 1.15, '\u2191 $\sigma$ = \u2193 Price Sensitivity', fontsize=20, va='top',
           bbox=dict(facecolor='none', edgecolor='lightgrey', boxstyle='round'))
    
    fig.text(0.52, 1.15, r'$\rho_c$ = Corr. of two correlated models', fontsize=20, va='top',
           bbox=dict(facecolor='none', edgecolor='lightgrey', boxstyle='round'))
    
    plt.subplots_adjust(right=0.97, wspace=0.05, hspace=0.15)
    plt.savefig('../figs/first_stage_game.pdf', bbox_inches='tight')
    plt.show()

params = [{'fixed': {'theta': 0.75, 'H': 3, 'L': 1, 'sigma': 0.1}, 'rho_corr': 1},
         {'fixed': {'theta': 0.75, 'H': 3, 'L': 1, 'sigma': 0.1}, 'rho_corr': 0.5},
         {'fixed': {'theta': 0.75, 'H': 3, 'L': 1, 'sigma': 0.3}, 'rho_corr': 1}]
mult(params)