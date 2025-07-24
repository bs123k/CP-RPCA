import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.linalg as scilinalg
import os
from SPG import *
from proj import *
from utils import *
import time
from numpy.linalg import norm
from joblib import Parallel, delayed

import matplotlib
matplotlib.use('TkAgg')

def gen_P(d1, d2, het, pr):
    # Generate the missing probability matrix P based on the heterogeneity parameters.
    if het=='rank1':
        lo_p = 0.5
        up_p = 0.9
        u = np.random.uniform(lo_p, up_p, d1).reshape((d1, 1))
        v = np.random.uniform(lo_p, up_p, d2).reshape((d2, 1))
        P = u @ v.T
    elif het == 'logis1':
        L = 0.5
        k_p = 5
        u = L * np.random.uniform(0, 2, d1 * k_p).reshape((d1, k_p))
        v = L * np.random.uniform(-1, 1, d2 * k_p).reshape((d2, k_p))
        AA = u @ v.T
        P = 1/(1+np.exp(-AA))
    elif het == 'logis2':
        L = 0.5
        k_l = 1
        u = L * np.random.uniform(0, 2, d1 * k_l).reshape((d1, k_l))
        v = L * np.random.uniform(-1, 1, d2 * k_l).reshape((d2, k_l))
        AA = u @ v.T
        P = 1/(1+np.exp(-AA))
    elif het == 'homo':
        P = pr * np.ones((d1, d2))
    return P


def generate_data(d1=1000, d2=1000, p=0.5, sparsity=0.1, r=20, missing_type='homo', noise_type='uniform'):
    # Generate low-rank matrices, sparse noise, observation masks, and observation data.
    U0 = np.random.randn(d1, r)/np.sqrt(r)
    V0 = np.random.randn(d2, r)/np.sqrt(r)

    U0, _ = np.linalg.qr(U0)
    V0, _ = np.linalg.qr(V0)
    X_star = U0 @ V0.T
    X_temp = X_star.copy()

    d1, d2 = U0.shape[0], V0.shape[0]
    P = gen_P(d1, d2, missing_type, p)
    assert 0 <= sparsity <= 1, "sparsity must be between [0,1]"
    assert np.all(P >= 0) and np.all(P <= 1), "The elements in P must be between [0,1]"

    # Generate observation position mask
    Omega = np.random.rand(d1, d2) < P
    supp = Omega.astype(float)

    I0, J0 = np.where(Omega)
    n_obs = len(I0)
    n_noise = int(sparsity * n_obs)
    if n_noise > 0:
        noise_indices = np.random.choice(n_obs, n_noise, replace=False)
    else:
        noise_indices = []

    if n_noise > 0:
        # Obtain the symbol of the original X at the noise position
        X_sign = np.sign(X_temp[I0[noise_indices], J0[noise_indices]])
        scaling_factor = r / np.sqrt(d1 * d2)
        # noise_S = (np.random.rand(n_noise) * 10 * scaling_factor) * X_sign
        noise_S = (np.random.rand(n_noise) * 10 * scaling_factor) * np.sign(np.random.randn(n_noise))

        X_temp[I0[noise_indices], J0[noise_indices]] = noise_S

    S_star = X_star - X_temp

    # Generate noise
    if noise_type == 'uniform':
        E = np.random.rand(d1, d2) * 0.1 * np.max(np.abs(X_star))  # uniformly distributed noise
    elif noise_type == 'normal':
        E =  0.1 * np.max(np.abs(X_star)) * np.random.randn(d1, d2)  # Gaussian distribution noise
    elif noise_type == 'laplace':
        E = 0.1 * np.max(np.abs(X_star)) * np.random.laplace(scale=1.0, size=(d1, d2))  # Laplace distribution noise
    elif noise_type == 't':
        E = 0.1 * np.max(np.abs(X_star)) * np.random.standard_t(3, size=(d1, d2))  # t Distributed noise
    elif noise_type == 'het':
        E = 0.1 * np.max(np.abs(X_star)) * np.random.normal(0, (0.5 / P).ravel(), d1 * d2).reshape((d1, d2))    # Adversarial heterogeneous noise
    elif noise_type == 'het1':
        Q = gen_P(d1, d2, missing_type, p)
        E = 0.1 * np.max(np.abs(X_star)) * np.random.normal(0, (0.5 / Q).ravel(), d1 * d2).reshape((d1, d2))   # random heterogeneous noise
    else:
        raise ValueError(f"Unsupported noise types: {noise_type}")

    lower_quantile = np.quantile(E, 0.05)
    upper_quantile = np.quantile(E, 0.95)
    quantile_diff = upper_quantile - lower_quantile

    X = X_star + E
    Y = X_star + S_star + E
    Y_obs = Ai(Y, supp)

    return X_star, S_star, X, Y, Y_obs, P, supp, quantile_diff


def rpca_gd(Y, O_train, p, params, X_star):
    step_const = params.get('step_const', 0.5)
    max_iter = params.get('max_iter', 60)
    r = params.get('r', 8)
    sparsity = params.get('sparsity', 0.1)
    tol = params.get('tol', 2e-4)
    incoh = params.get('incoh', 5)
    gamma = params.get('gamma', 1)
    do_project = params.get('do_project', False)
    L = params.get('L', 5)
    tau_prime = params.get('tau_prime', 0.1)
    eta_prime = params.get('eta_prime', 0.1)
    verbose = params.get('verbose', True)
    f_grad = lambda X: -1 / p * (Y - (X * O_train))

    err_hist = []
    time_hist = []
    x_errors = []
    start_time = time.time()

    Ynormfro = np.linalg.norm(Y, 'fro')
    d1, d2 = Y.shape
    n = d1 * d2

    alpha_col = sparsity
    alpha_row = sparsity
    S = Tproj_partial(Y, gamma * p * alpha_col, gamma * p * alpha_row)

    M_init = (Y - S) / p
    try:
        U_svd, s_svd, Vt_svd = svds(M_init, k=r, which='LM')
        idx = np.argsort(s_svd)[::-1]
        s_svd = s_svd[idx]
        U_svd = U_svd[:, idx]
        Vt_svd = Vt_svd[idx, :]
    except Exception as e:
        U_full, s_full, Vt_full = np.linalg.svd(M_init, full_matrices=False)
        U_svd = U_full[:, :r]
        s_svd = s_full[:r]
        Vt_svd = Vt_full[:r, :]

    U = U_svd * np.sqrt(s_svd)
    V = Vt_svd.T * np.sqrt(s_svd)
    X = U @ V.T

    for ell in range(L):
        grad_S = f_grad(X + S)
        S_temp = S - tau_prime * grad_S
        S_new = HT(S_temp, k=int(gamma * sparsity * p * n))
        grad_X = f_grad(X + S_new)
        X_temp = X - eta_prime * grad_X
        X_new = X_temp
        X = np.nan_to_num(X_new, nan=0.0, posinf=1e6, neginf=-1e6)
        S = np.nan_to_num(S_new, nan=0.0, posinf=1e6, neginf=-1e6)
        if verbose:
            error = np.linalg.norm(Y - (X + S) * O_train, 'fro') / np.sqrt(n) / p
            # print(f"Init Iter {ell}: Obs MSE={error:.8e}")

    U0, s0, Vt0 = svds(X, k=r)
    U = U0 @ np.diag(np.sqrt(s0))
    V = Vt0.T @ np.diag(np.sqrt(s0))

    err_hist.append(error)
    time_hist.append(time.time() - start_time)

    mu, sigma1, cond_num = calc_para(U @ V.T, r)
    Z0_norm = np.linalg.norm(np.vstack([U, V]), 2)

    steplength = step_const / s_svd[0] / r / mu * 2.5
    converged = False
    iter_num = 0
    eps_val = 1e-8
    while not converged:
        iter_num += 1

        YminusUV = (Y - U @ V.T) * O_train

        S = Tproj_partial(YminusUV, gamma * p * alpha_col, gamma * p * alpha_row)
        E = YminusUV - S

        U_new = U + steplength * (E @ V) / p - (steplength / 16) * U @ ((U.T @ U) - (V.T @ V))
        V_new = V + steplength * (E.T @ U) / p - (steplength / 16) * V @ ((V.T @ V) - (U.T @ U))

        if do_project:
            const1 = np.sqrt(4 * mu * r / d1) * Z0_norm
            const2 = np.sqrt(4 * mu * r / d2) * Z0_norm
            U_norm = np.linalg.norm(U_new, axis=1, keepdims=True)
            U_new = U_new * np.minimum(1, const1 / (U_norm + 1e-10))
            V_norm = np.linalg.norm(V_new, axis=1, keepdims=True)
            V_new = V_new * np.minimum(1, const2 / (V_norm + 1e-10))

        U = U_new
        V = V_new

        X_hat = U @ V.T
        x_error = np.linalg.norm(X_hat - X_star, 'fro') / np.sqrt(d1 * d2)
        x_errors.append(x_error)
        err = np.linalg.norm(E, 'fro') / Ynormfro
        err_hist.append(err)
        time_hist.append(time.time() - start_time)

        if iter_num >= max_iter:
            # print("Reached maximum iteration count")
            converged = True
        elif err <= max(tol, eps_val):
            # print("Achieve target error")
            converged = True

    return U, V, err_hist, time_hist, x_errors

def rpca_alm(D, mask, params):
    # RPCA via Augmented Lagrange Multiplier (ALM) for partially observed data.

    max_iter = params.get('max_iter', 200)
    tol = params.get('tol', 1e-7)     # Convergence tolerance
    lambda_param = params.get('lambda_param', None)       # Regularization parameter (default: 1/sqrt(max(m,n)))

    m, n = D.shape
    if lambda_param is None:
        lambda_param = 1 / np.sqrt(max(m, n))

    # Initialize variables
    D = np.nan_to_num(D, nan=0.0, posinf=1e10, neginf=-1e10)
    X = np.zeros((m, n))
    S = np.zeros((m, n))
    Y = np.zeros((m, n))  # Lagrange multiplier
    mu = 0.25 / (norm(D * mask, 'fro') + 1e-10)  # Initial mu
    rho = 1.1  # Update factor for mu
    errors = []

    for k in range(max_iter):
        # Update L (low-rank matrix)
        X = svd_threshold(D - S + (1 / mu) * Y, 1 / mu)

        # Update S (sparse matrix)
        S_temp = D - X + (1 / mu) * Y
        S = soft_threshold(S_temp, lambda_param / mu)
        S = mask * S  # Apply mask to S

        # Update Lagrange multiplier Y
        Z = D - X - S
        Z = mask * Z  # Apply mask to residual
        Y = Y + mu * Z

        # Compute residual
        residual = norm(Z, 'fro') / (norm(D * mask, 'fro') + 1e-10)
        errors.append(residual)

        # Check convergence
        if residual < tol:
            break

        # Update penalty parameter mu
        mu = min(rho * mu, 1e10)

    return X, S, errors


def cp_rpca(Y, X_star, S_star, supp, P, params):

    q = params['q']
    missing_model = params['missing_model']
    alpha = params['alpha']
    Y_obs = Y * supp
    d1, d2 = Y_obs.shape
    n_obs = np.sum(supp)
    supp_bool = supp.astype(bool)
    p = params['p']

    # Divide the data into training and validation sets
    train_selector = np.less(np.random.rand(d1, d2), q)
    O_train = (supp * train_selector).astype(bool)
    O_cal = (supp * (1 - train_selector)).astype(bool)
    n_train = np.sum(O_train)
    n_cal = int(n_obs - n_train)
    train_coords = np.argwhere(O_train)
    assert n_cal == np.sum(O_cal)
    Y_train = Y_obs * O_train
    Y_cal = Y_obs * O_cal

    P_hat = estimate_P(O_train, q, P, missing_model)

    fast_rpca_params = {
        'gamma': 1.2,
        'sparsity': params['sparsity'],
        'step_const': 0.5,
        'max_iter': params['max_iter'],
        'tol': 1e-8,
        'incoh': 5,
        'do_project': False,
        'r': params['r']
    }

    n_bootstrap = 30

    def bootstrap_iteration(seed):
        np.random.seed(seed)
        selected_indices = np.random.choice(n_train, size=n_train, replace=True)
        selected_coords = train_coords[selected_indices]

        O_train_boot = np.zeros_like(O_train, dtype=bool)
        O_train_boot[selected_coords[:, 0], selected_coords[:, 1]] = True

        Y_train_boot = Y_obs * O_train_boot
        p_temp = np.sum(O_train_boot) / O_train_boot.size

        if params.get('robust_pca_method', 'fast_rpca') == 'fast_rpca':
            U, V, err_hist, time_hist, x_errors = rpca_gd(Y_train_boot, O_train_boot, p_temp, fast_rpca_params, X_star)
            X_hat = U @ V.T
        elif params.get('robust_pca_method', 'fast_rpca') == 'rpca_alm':
            X_hat, S, _ = rpca_alm(Y_train_boot, O_train_boot, params={'max_iter': params['max_iter'],'tol': 1e-8})
        else:
            raise ValueError("Unknown robust PCA method")

        return X_hat

    seeds = np.random.randint(0, 1e6, size=n_bootstrap)

    # Parallel execution of Bootstrap
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(bootstrap_iteration)(seed) for seed in seeds
    )

    X_bootstrap_list = results

    X_hat_mean = np.mean(X_bootstrap_list, axis=0)
    X_std = np.std(X_bootstrap_list, axis=0, ddof=1)

    if params.get('robust_pca_method', 'fast_rpca') == 'fast_rpca':
        U, V, _, _, _ = rpca_gd(Y_train, O_train, p, fast_rpca_params, X_star)
        X_hat = U @ V.T
    elif params.get('robust_pca_method', 'fast_rpca') == 'rpca_alm':
        X_hat, S, _ = rpca_alm(Y_train, O_train, params={'max_iter': params['max_iter'], 'tol': 1e-8})

    gamma = 1.1
    S = Tproj_partial((Y_obs - X_hat) * supp, gamma * p * params['sparsity'],
                      gamma * p * params['sparsity'])
    mask = np.logical_and(Y_train, (S == 0))
    count = np.sum(mask)

    sigma = np.sqrt(np.sum(((X_hat - Y_obs) * mask) ** 2) / count)

    s_hat = np.sqrt(X_std ** 2 + sigma ** 2)

    # Calculate calibration score
    score = np.abs(Y_obs - X_hat) / (s_hat)

    observation_matrix = (S[O_cal] == 0).astype(bool)
    m = np.sum(observation_matrix)

    # Calculate weight H_hat
    H_hat = (1 - P_hat) / P_hat
    w_max = np.max((1 - supp) * H_hat)
    ww = np.zeros(int(m) + 1)
    ww[:int(m)] = H_hat[O_cal][observation_matrix].ravel()
    ww[int(m)] = w_max

    r_new = 10000
    r_all = np.append(score[O_cal][observation_matrix], r_new)
    qvals = weighted_quantile(r_all, prob=1 - alpha, w=ww)

    lo_mat = X_hat - qvals * s_hat
    up_mat = X_hat + qvals * s_hat

    lo = lo_mat[~supp_bool].reshape(-1)
    up = up_mat[~supp_bool].reshape(-1)
    return lo, up, r_all, qvals, X_hat


def run_rank_experiment(rank_range, params_base, data_params):
    """Running conformal prediction experiments under different rank assumptions"""

    coverage_list = []
    length_list = []

    for r_hat in rank_range:
        # print(f"Running assumption rank r={r_hat}")
        alg_params = params_base.copy()
        alg_params['r'] = r_hat
        lo, up, _, _, _, = cp_rpca(
            data_params['Y'], data_params['X_star'], data_params['S_star'],
            data_params['supp'], data_params['P'], alg_params
        )
        supp_bool = data_params['supp'].astype(bool)

        matrix = data_params['Y']
        x_true = matrix[~supp_bool].reshape(-1)
        coverage = np.mean((lo <= x_true) & (x_true <= up))
        avg_length = np.mean(up - lo)

        coverage_list.append(coverage)
        length_list.append(avg_length)

    return list(coverage_list), list(length_list)


def run_experiment():
    base_seed = 12345
    d1, d2 = 500, 500
    true_rank = 8
    sparsity = 0.1
    p = 0.5
    q = 0.6
    rank_range = np.arange(4, 17, 2)
    n_sim = 30

    scenarios = [
        {'missing_type': 'homo', 'noise_type': 'normal', 'missing_model': 'homo', 'p': 0.5, 'sparsity': 0.1,
         'label': 'Uniform observation + Gaussian noise'},
        {'missing_type': 'homo', 'noise_type': 't', 'missing_model': 'homo', 'p': 0.5, 'sparsity': 0.1,
         'label': 'Uniform observation + heavy-tailed noise'},
        {'missing_type': 'rank1', 'noise_type': 'normal', 'missing_model': 'rank1', 'p': 0.5, 'sparsity': 0.1,
         'label': 'Rank-1 missing + Gaussian noise'},
        {'missing_type': 'rank1', 'noise_type': 't', 'missing_model': 'rank1', 'p': 0.5, 'sparsity': 0.1,
         'label': 'Rank-1 missing + heavy-tailed noise'},
        {'missing_type': 'logis1', 'noise_type': 'normal', 'missing_model': 'logis1', 'p': 0.5, 'sparsity': 0.1,
         'label': 'Logistic missing + Gaussian noise'},
        {'missing_type': 'logis1', 'noise_type': 't', 'missing_model': 'logis1', 'p': 0.5, 'sparsity': 0.1,
         'label': 'Logistic missing + heavy-tailed noise'}
    ]
    '''scenarios = [
        {'missing_type': 'logis1', 'noise_type': 'het', 'missing_model': 'logis1', 'p': 0.5, 'sparsity': 0.1,
         'label': 'logistic missing + \nAdversarial heterogeneous noise'},
        {'missing_type': 'logis1', 'noise_type': 'het1', 'missing_model': 'logis1', 'p': 0.5, 'sparsity': 0.1,
         'label': 'logistic missing + \nRandom heterogeneous noise'},
    ]'''
    '''scenarios = [
        {'missing_type': 'homo', 'noise_type': 'normal', 'missing_model': 'homo', 'p': 0.5, 'sparsity': 0.05,
         'label': 'Low sparsity + Gaussian noise'},
        {'missing_type': 'homo', 'noise_type': 'normal', 'missing_model': 'homo', 'p': 0.5, 'sparsity': 0.2,
         'label': 'High sparsity + Gaussian noise'},
    ]'''

    # 创建整体画布，3行2列，每行两种模式
    fig = plt.figure(figsize=(16, 8))
    outer_grid = gridspec.GridSpec(3, 2, figure=fig, wspace=0.15, hspace=0.45)

    for idx, scenario in enumerate(scenarios):
        row = idx // 2
        col = idx % 2

        # 每个模式内部的小网格
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[row, col], wspace=0.27)
        ax1 = plt.Subplot(fig, inner_grid[0])  # 覆盖率子图
        ax2 = plt.Subplot(fig, inner_grid[1])  # 长度子图

        # 初始化存储结果
        all_coverage = []
        all_length = []
        diff = []

        for _ in range(n_sim):
            base_seed += 1
            np.random.seed(base_seed)
            X_star, S_star, X, Y, Y_obs, P, supp, quantile_diff = generate_data(
                d1=d1, d2=d2, p=scenario['p'], sparsity=scenario['sparsity'], r=true_rank,
                missing_type=scenario['missing_type'],
                noise_type=scenario['noise_type']
            )

            p_val = np.mean(supp)
            mu, sigma1, cond_num = calc_para(X_star, true_rank)

            params = {
                'missing_model': scenario['missing_model'],
                'alpha': 0.1,         # expected coverage of 1-alpha
                'robust_pca_method': 'fast_rpca',
                'p': p_val,
                'q': q,
                'sparsity': scenario['sparsity'],
                'mu': mu,
                'step_const': 0.5,
                'max_iter': 120,
                'tol': 1e-10,
                'incoh': 5,
                'do_project': True
            }
            data_params = {
                'Y_obs': Y_obs,
                'X_star': X_star,
                'S_star': S_star,
                'supp': supp,
                'Y': Y,
                'P': P,
                'X': X
            }

            params['robust_pca_method'] = 'fast_rpca'
            coverage, length = run_rank_experiment(
                rank_range, params,
                data_params=data_params
            )

            all_coverage.append(coverage)
            all_length.append(length)
            diff.append(quantile_diff)


        # 计算平均结果
        avg_coverage = np.mean(all_coverage, axis=0)
        avg_length = np.mean(all_length, axis=0)
        avg_diff = np.mean(diff, axis=0)

        ax1.plot(rank_range, avg_coverage, 'b-o', linewidth=2, markersize=4)
        ax1.axhline(0.9, color='r', linestyle='--', label='Target Coverage Rate')
        ax1.axvline(true_rank, color='g', linestyle=':', linewidth=3)
        ax1.set_ylim(0.6, 1)
        ax1.set_xlabel('hypothesized rank', fontsize=9)
        ax1.set_ylabel('AvgCov', fontsize=9)
        ax1.set_title(f'{scenario["label"]}', fontsize=10)
        ax1.tick_params(axis='both', labelsize=9)
        ax1.grid(True)
        ax1.legend(fontsize=8, labelspacing=0.3)

        ax2.plot(rank_range, avg_length, 'm-s', linewidth=2, markersize=4)
        if scenario['noise_type'] not in ['het', 'het1']:
            ax2.axhline(avg_diff, color='r', linestyle='--',
                        label=f'oracle')
            # ({avg_diff:.3f})
            ax2.legend(fontsize=8, labelspacing=0.3)
        ax2.axvline(true_rank, color='g', linestyle=':', linewidth=3)
        ax2.set_ylim(0.008, 0.03)
        ax2.set_xlabel('hypothesized rank', fontsize=9)
        ax2.set_ylabel('AvgLength', fontsize=9)
        ax2.set_title(f'{scenario["label"]}', fontsize=10)
        ax2.tick_params(axis='both', labelsize=9)
        ax2.grid(True)

        fig.add_subplot(ax1)
        fig.add_subplot(ax2)

    plt.tight_layout()
    save_dir = "../images"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "CP_RPCA_results.svg"), format='svg', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run_experiment()
