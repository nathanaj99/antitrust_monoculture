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
import scipy.stats as stats
from scipy.optimize import brentq
import itertools
import sympy as sp

cmap = sns.diverging_palette(20, 255, s=100, l=35, as_cmap=True, center='light')


def get_vals(expr, x, y, x_var, y_var, vals):
    t = expr.subs(vals)
    f = lambdify([x_var, y_var], t, 'numpy')
    
    res = f(x, y)
    return res

def binary_to_gaussian_correlation(rho_binary, p=0.5):
    """
    Convert binary correlation (rho_binary) to Gaussian correlation (rho_gaussian)
    using a numerical approximation.

    Parameters:
    - rho_binary: float, binary correlation
    - p: float, marginal probability of success (default is 0.5 for symmetric Bernoulli)

    Returns:
    - rho_gaussian: float, corresponding Gaussian correlation
    """
    if rho_binary == 0:
        return 0.0

    # Thresholds for binary outcomes
    t = stats.norm.ppf(p)

    def binary_corr_from_gaussian(rho_gaussian):
        """Calculate binary correlation given Gaussian correlation."""
        joint_cdf = stats.multivariate_normal.cdf(
            [t, t],
            mean=[0, 0],
            cov=[[1, rho_gaussian], [rho_gaussian, 1]]
        )
        return (joint_cdf - p * p) / np.sqrt(p * (1 - p) * p * (1 - p))

    # Solve numerically for rho_gaussian
    try:
        rho_gaussian = brentq(
            lambda rho: binary_corr_from_gaussian(rho) - rho_binary,
            -0.999, 0.999
        )
    except ValueError:
        raise ValueError("No valid solution found for rho_gaussian. Check input rho_binary.")
    
    return rho_gaussian


def build_correlation_matrix(n, rho_g, cluster_correlations, cluster_indices):
    """
    Build the Gaussian correlation matrix.
    """
    rho_g_gaussian = binary_to_gaussian_correlation(rho_g)
    cluster_correlations_gaussian = [binary_to_gaussian_correlation(rho_c) for rho_c in cluster_correlations]

    corr_matrix = np.eye(n)

    # Intra-cluster correlations
    for cluster, rho_c in zip(cluster_indices, cluster_correlations_gaussian):
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                corr_matrix[cluster[i], cluster[j]] = rho_c
                corr_matrix[cluster[j], cluster[i]] = rho_c

    # Inter-cluster correlations
    for i in range(n):
        for j in range(n):
            if corr_matrix[i, j] == 0 and i != j:
                corr_matrix[i, j] = rho_g_gaussian
                corr_matrix[j, i] = rho_g_gaussian

    # Ensure positive semi-definiteness
    eigvals = np.linalg.eigvals(corr_matrix)
    if np.any(eigvals < 0):
        raise ValueError("Correlation matrix is not positive semi-definite. Check your parameters.")
    
    return corr_matrix


def compute_joint_probabilities(n, corr_matrix, accuracies, tau):
    """
    Compute exact joint probabilities P(p_1, ..., p_n | tau)
    using the Gaussian copula with player-specific accuracies.

    Parameters:
    - n: int, number of players
    - corr_matrix: numpy.ndarray, Gaussian correlation matrix
    - accuracies: list of floats, marginal probability for each player

    Returns:
    - joint_probs: dict, keys are tuples (e.g., (1,0,1)) and values are probabilities
    """
    if len(accuracies) != n:
        raise ValueError("Length of accuracies must match the number of players (n).")

    # Thresholds based on accuracies
    if tau == "H":
        thresholds = [stats.norm.ppf(a) for a in accuracies]
    else:
        thresholds = [stats.norm.ppf(1-a) for a in accuracies]

    # Thresholds based on tau
    # p_success = 0.8 if tau == "H" else 0.2  # Example probabilities for H and L
    # thresholds = stats.norm.ppf(p_success) * np.ones(n)

    # Iterate through all binary outcomes
    outcomes = list(itertools.product([0, 1], repeat=n))
    joint_probs = {}

    for outcome in outcomes:
        bounds = []
        for i, bit in enumerate(outcome):
            if bit == 1:
                bounds.append((-np.inf, thresholds[i]))
            else:
                bounds.append((thresholds[i], np.inf))

        # Compute multivariate normal CDF for the given bounds
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]

        prob = stats.mvn.mvnun(lower_bounds, upper_bounds, np.zeros(n), corr_matrix)[0]
        joint_probs[outcome] = prob

    return joint_probs

def fully_correlated_probabilities(accuracies):
    # Create all possible binary combinations as keys
    n = len(accuracies)
    all_combinations = list(itertools.product([0, 1], repeat=n))

    # Initialize dictionaries with zeros
    P_H = {combo: 0.0 for combo in all_combinations}
    P_L = {combo: 0.0 for combo in all_combinations}

    # Set values for extreme cases
    all_ones = tuple([1] * n)
    all_zeros = tuple([0] * n)

    # For tau_H:
    # P(all 1s) = min(accuracies)
    # P(all 0s) = 1 - min(accuracies)
    P_H[all_ones] = min(accuracies)
    P_H[all_zeros] = 1 - min(accuracies)

    # For tau_L:
    # P(all 1s) = 1 - max(accuracies)
    # P(all 0s) = max(accuracies)
    P_L[all_ones] = 1 - min(accuracies)
    P_L[all_zeros] = min(accuracies)

    return P_H, P_L


# Symbolic parameters
H, L, sigma, theta, a = sp.symbols('H L sigma theta a', real=True, positive=True)


# Define utility function
def utility_function(tau, all_same, strategy, n_h, n_l):
    if tau == 'tau_H':
        if all_same:
            return H / n if strategy == 'H' else L / n
        else:
            return sigma * H / n_h if strategy == 'H' else (1 - sigma) * L / n_l
    elif tau == 'tau_L':
        if all_same:
            return 0 if strategy == 'H' else L / n
        else:
            return 0 if strategy == 'H' else L / n_l

# Compute expected utility numerically
def expected_utility_numeric(player_idx, n_players, P_H, P_L, strategy, p_i):
    """
    Compute E[U_i(strategy, s^*(-i)) | p_i = 1] numerically using precomputed probabilities.

    Parameters:
    - player_idx: int, index of the player i
    - n_players: int, total number of players
    - P_H: dict, joint probabilities for tau_H
    - P_L: dict, joint probabilities for tau_L

    Returns:
    - float, expected utility for player i
    """
    outcomes = list(itertools.product([0, 1], repeat=n_players - 1))
    EU = 0

    for outcome in outcomes:
        # Create a full outcome tuple with p_i = 1
        full_outcome = list(outcome)
        full_outcome.insert(player_idx, p_i)  # Insert p_i = 1 at the correct index
        full_outcome = tuple(full_outcome)
        
        # Count high (H) and low (L) plays among other players
        n_h = sum(outcome)
        n_l = (n_players - 1) - n_h

        # add one for player i, depending on their strategy
        if strategy == 'H':
            n_h += 1
        else:
            n_l += 1
        
        all_same = (n_h == 0 or n_l == 0)
        # Compute utilities for both states
        U_H = utility_function('tau_H', all_same, strategy, n_h, n_l)
        U_L = utility_function('tau_L', all_same, strategy, n_h, n_l)
        
        # Get probabilities from precomputed dictionaries
        P_H_val = P_H.get(full_outcome, 0)
        P_L_val = P_L.get(full_outcome, 0)
        
        # Weighted sum
        EU += theta * U_H * P_H_val + (1 - theta) * U_L * P_L_val
    
    return EU

def eq_conditions_byplayer(n, player_idx, P_H, P_L):
    # player_idx = 0
    u_h_1 = expected_utility_numeric(player_idx, n, P_H, P_L, 'H', 1)
    u_l_1 = expected_utility_numeric(player_idx, n, P_H, P_L, 'L', 1)
    u_h_0 = expected_utility_numeric(player_idx, n, P_H, P_L, 'H', 0)
    u_l_0 = expected_utility_numeric(player_idx, n, P_H, P_L, 'L', 0)

    ex0 = u_h_0 - u_l_0
    ex1 = u_h_1 - u_l_1

    return ex0, ex1, u_h_1, u_l_1, u_h_0, u_l_0

def get_probs(n, k, acc):
    if k == 1:
        cluster_correlations = [0 for i in range(n)]
    elif k == n:
        cluster_correlations = [1]
    else:
        cluster_correlations = [0.95] + [0 for i in range(k, n)]
    cluster_indices = [list(range(k))] + [[i] for i in range(k, n)]
    accuracies = [acc] * n
    fully_correlated = True if cluster_correlations[0] == 1 else False
    if fully_correlated:
        P_H, P_L = fully_correlated_probabilities(accuracies)
    else:
        # Build correlation matrix
        corr_matrix = build_correlation_matrix(n, rho_g, cluster_correlations, cluster_indices)

        # Compute joint probabilities for tau_H and tau_L
        P_H = compute_joint_probabilities(n, corr_matrix, accuracies, tau="H")
        P_L = compute_joint_probabilities(n, corr_matrix, accuracies, tau="L")

    return P_H, P_L


x,y = np.meshgrid(np.linspace(0, 0.5, 501), np.linspace(0.5, 1, 1001)[:-1])
# prop = 5
# acc = 0.9
max_n = 5

prop_acc_list = [(3, 0.9), (5, 0.9), (7, 0.9), (5, 0.5), (5, 0.7)]
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(nrows=max_n-1, ncols=5, sharey=True, sharex=True, figsize=(15, 13))
for i, (prop, acc) in enumerate(prop_acc_list):
    for j, n in enumerate(range(2, max_n+1)):
        vals = {H: prop, L: 1}
        # generate k = 0, ..., n. k = 0 all independent, k = n all correlated
        for k, color in zip(range(n+1), ['lightgray', 'lightblue', 'lightgreen', 'orange', 'red', 'purple']):
            rho_g = 0
            P_H, P_L = get_probs(n, k, acc)

            # second-stage game
            res = np.ones((x.shape[0], x.shape[1]), dtype=bool)
            for player_idx in range(n):
                # second-stage game
                ex0, ex1, u_h_1, u_l_1, u_h_0, u_l_0 = eq_conditions_byplayer(n, player_idx, P_H, P_L)

                res0 = get_vals(ex0 <= 0, x, y, sigma, theta, vals)
                res1 = get_vals(ex1 >= 0, x, y, sigma, theta, vals)
                res = res & res0 & res1

            # first-stage game
            if k == 1: # all independent, meaning that deviations are not correlated
                P_H_alt, P_L_alt = get_probs(n, k+1, acc)
                player_idx = k
                ex0, ex1, u_h_1, u_l_1, u_h_0, u_l_0 = eq_conditions_byplayer(n, player_idx, P_H, P_L)
                ex0_alt, ex1_alt, u_h_1_alt, u_l_1_alt, u_h_0_alt, u_l_0_alt = eq_conditions_byplayer(n, player_idx, P_H_alt, P_L_alt)
                condition = get_vals(u_h_1 + u_l_0 >= u_h_1_alt + u_l_0_alt, x, y, sigma, theta, vals)
                res = res & condition

            elif k == n: # all correlated
                P_H_alt, P_L_alt = get_probs(n, k-1, acc)
                # sufficient to check last player that deviates, as per probabilities above
                player_idx = n-1
                ex0, ex1, u_h_1, u_l_1, u_h_0, u_l_0 = eq_conditions_byplayer(n, player_idx, P_H, P_L)
                ex0_alt, ex1_alt, u_h_1_alt, u_l_1_alt, u_h_0_alt, u_l_0_alt = eq_conditions_byplayer(n, player_idx, P_H_alt, P_L_alt)

                condition = get_vals(u_h_1 + u_l_0 >= u_h_1_alt + u_l_0_alt, x, y, sigma, theta, vals)
                res = res & condition
            elif k == 0:
                pass
            else:
                # first, check if player in coalition is stable. sufficient to check one player
                P_H_alt, P_L_alt = get_probs(n, k-1, acc)
                player_idx = k-1
                ex0, ex1, u_h_1, u_l_1, u_h_0, u_l_0 = eq_conditions_byplayer(n, player_idx, P_H, P_L)
                ex0_alt, ex1_alt, u_h_1_alt, u_l_1_alt, u_h_0_alt, u_l_0_alt = eq_conditions_byplayer(n, player_idx, P_H_alt, P_L_alt)
                condition = get_vals(u_h_1 + u_l_0 >= u_h_1_alt + u_l_0_alt, x, y, sigma, theta, vals)
                res = res & condition

                # then, check if player outside coalition is stable. sufficient to check one player
                P_H_alt, P_L_alt = get_probs(n, k+1, acc)
                player_idx = k
                ex0, ex1, u_h_1, u_l_1, u_h_0, u_l_0 = eq_conditions_byplayer(n, player_idx, P_H, P_L)
                ex0_alt, ex1_alt, u_h_1_alt, u_l_1_alt, u_h_0_alt, u_l_0_alt = eq_conditions_byplayer(n, player_idx, P_H_alt, P_L_alt)
                condition1 = get_vals(u_h_1 + u_l_0 >= u_h_1_alt + u_l_0_alt, x, y, sigma, theta, vals)
                res = res & condition1

            ax[j, i].imshow((res).astype(int),
                    origin="lower", cmap=ListedColormap(['none', color]),
                extent=(x.min(),x.max(),y.min(),y.max()), aspect='auto', alpha=0.5, label=k)
        if j == 0: 
            ax[j, i].set_title(r'$H/L$ = {}'.format(prop) + '\n' + r'$a$ = {}'.format(acc))
        if j == 3: 
            ax[j, i].set_xlabel(r'$\sigma$')
        ax[j, i].set_xticks([0, 0.2, 0.4])
        if i == 0:
            ax[j, i].set_ylabel(r'$\theta$')


# 'lightgray', 'lightblue', 'lightgreen', 'lightyellow', 'red'
# Define custom patches for the legend
custom_patches = [
    Patch(color='lightgray', label='k=0', alpha=0.5),
    Patch(color='lightblue', label='k=1', alpha=0.5),
    Patch(color='lightgreen', label='k=2', alpha=0.5),
    Patch(color='orange', label='k=3', alpha=0.5),
    Patch(color='red', label='k=4', alpha=0.5),
    Patch(color='purple', label='k=5', alpha=0.5),
]

rows = [r'$n$ = {}'.format(row) for row in range(2, max_n+1)]

pad = 5 # in points

for aa, row in zip(ax[:,0], rows):
    aa.annotate(row, xy=(0, 0.5), xytext=(-aa.yaxis.labelpad - pad, 0),
                xycoords=aa.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

# Add the legend to the figure
fig.legend(handles=custom_patches, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=max_n+1)

fig.text(0.03, 1.0, '\u2191 $\sigma$ = \u2193 Price Sensitivity', fontsize=20, va='top',
           bbox=dict(facecolor='none', edgecolor='lightgrey', boxstyle='round'))
    
fig.text(0.31, 1.0, '$H/L$ = Ratio of High to Low Price', fontsize=20, va='top',
        bbox=dict(facecolor='none', edgecolor='lightgrey', boxstyle='round'))

fig.text(0.64, 1.0, r'$\theta$ = % Population Willing to Pay $H$', fontsize=20, va='top',
        bbox=dict(facecolor='none', edgecolor='lightgrey', boxstyle='round'))

plt.subplots_adjust(right=0.98, wspace=0.1, hspace=0.15)
plt.savefig('../figs/n_player_game.pdf', bbox_inches='tight')
plt.show()