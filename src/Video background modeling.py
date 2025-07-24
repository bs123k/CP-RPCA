import numpy as np
import cv2
import os, glob
import matplotlib.pyplot as plt
import time
import os
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from joblib import Parallel, delayed
from utils import *

import matplotlib
matplotlib.use('TkAgg')

def Tproj_partial(A, tau_col, tau_row):
    n_rows, n_cols = A.shape
    mask_col = np.zeros_like(A, dtype=bool)
    mask_row = np.zeros_like(A, dtype=bool)
    allowed_col = int(np.ceil(tau_col * n_rows))
    allowed_row = int(np.ceil(tau_row * n_cols))

    # Process columns
    for j in range(n_cols):
        col_abs = np.abs(A[:, j])
        if allowed_col < n_rows:
            threshold = np.partition(col_abs, -allowed_col)[-allowed_col]
            mask_col[:, j] = col_abs >= threshold
        else:
            mask_col[:, j] = True

    # Process rows
    for i in range(n_rows):
        row_abs = np.abs(A[i, :])
        if allowed_row < n_cols:
            threshold = np.partition(row_abs, -allowed_row)[-allowed_row]
            mask_row[i, :] = row_abs >= threshold
        else:
            mask_row[i, :] = True

    mask = mask_col & mask_row
    return A * mask

# Compute observed entries of the low-rank matrix X = U @ V.T at indices (I, J)
def compute_X_Omega(U, V, I, J):
    return np.sum(U[I, :] * V[J, :], axis=1)

# Robust PCA using gradient descent
def rpca_gd(Y, r, alpha, params):
    step_const = params.get('step_const', 0.5)
    max_iter = params.get('max_iter', 30)
    tol = params.get('tol', 2e-4)
    do_project = params.get('do_project', False)

    err_hist = []
    time_hist = []
    start_time = time.time()

    if sp.issparse(Y):
        Y_dense = Y.toarray()
    else:
        Y_dense = Y
    Ynormfro = np.linalg.norm(Y_dense, 'fro')
    d1, d2 = Y_dense.shape

    is_sparse = sp.issparse(Y)
    if is_sparse:
        Y_coo = Y.tocoo()
        I = Y_coo.row
        J = Y_coo.col
        Y_vec = Y_coo.data
        n = len(Y_vec)
        p = n / (d1 * d2)
        if p > 0.9:
            is_sparse = False
            Y_dense = Y.toarray()
            p = 1
    else:
        p = 1

    # Initialize Phase I
    alpha_col = alpha
    alpha_row = alpha
    S = Tproj_partial(Y_dense, p * alpha_col, p * alpha_row)

    M_init = (Y_dense - S) / p
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

    if do_project:
        const1 = np.sqrt(4 * r / d1) * s_svd[0]
        const2 = np.sqrt(4 * r / d2) * s_svd[0]
        U_norm = np.linalg.norm(U, axis=1, keepdims=True)
        scaling_U = np.minimum(1, const1 / (U_norm + 1e-10))
        U = U * scaling_U
        V_norm = np.linalg.norm(V, axis=1, keepdims=True)
        scaling_V = np.minimum(1, const2 / (V_norm + 1e-10))
        V = V * scaling_V

    err = float('inf')
    err_hist.append(err)
    time_hist.append(time.time() - start_time)

    # Phase II: Gradient Descent
    steplength = step_const / s_svd[0]
    iter_num = 0
    eps_val = np.finfo(float).eps
    converged = False
    while not converged:
        iter_num += 1
        if is_sparse:
            UV_obs = compute_X_Omega(U, V, I, J)
            diff_data = Y_vec - UV_obs
            YminusUV = sp.coo_matrix((diff_data, (I, J)), shape=(d1, d2)).toarray()
        else:
            YminusUV = Y_dense - U @ V.T

        S = Tproj_partial(YminusUV, p * alpha_col, p * alpha_row)
        E = YminusUV - S

        U_new = U + steplength * (E @ V) / p - (steplength / 16) * U @ ((U.T @ U) - (V.T @ V))
        V_new = V + steplength * (E.T @ U) / p - (steplength / 16) * V @ ((V.T @ V) - (U.T @ U))

        if do_project:
            U_norm = np.linalg.norm(U_new, axis=1, keepdims=True)
            U_new = U_new * np.minimum(1, const1 / (U_norm + 1e-10))
            V_norm = np.linalg.norm(V_new, axis=1, keepdims=True)
            V_new = V_new * np.minimum(1, const2 / (V_norm + 1e-10))

        U = U_new
        V = V_new

        err = np.linalg.norm(E, 'fro') / Ynormfro
        err_hist.append(err)
        time_hist.append(time.time() - start_time)

        if iter_num >= max_iter:
            converged = True
        elif err <= max(tol, eps_val):
            converged = True
        elif err >= err_hist[-2] - eps_val:
            converged = True

    return U, V, S, err_hist, time_hist

def Ai(X, supp):
    return X.multiply(supp) if sp.issparse(X) else X * supp


# Conformalized RPCA algorithm
def cp_rpca(Y, supp, r, sparsity, params, alpha, q):
    d1, d2 = Y.shape
    n_obs = np.sum(supp)
    Y_obs = Y * supp

    # Split into training and calibration sets
    train_selector = np.less(np.random.rand(d1, d2), q)
    O_train = (train_selector * supp).astype(bool)
    O_cal = (supp * (1 - train_selector)).astype(bool)
    n_train = np.sum(O_train)
    n_cal = n_obs - n_train
    train_coords = np.argwhere(O_train).astype(np.int32)
    assert n_cal == np.sum(O_cal)
    Y_train = Y_obs * O_train
    p = np.mean(supp)

    n_bootstrap = 20

    def bootstrap_iteration(seed):
        np.random.seed(seed)  # Ensure independent random state for each process
        selected_indices = np.random.choice(n_train, size=n_train, replace=True)
        selected_coords = train_coords[selected_indices]

        O_train_boot = np.zeros_like(O_train, dtype=bool)
        O_train_boot[selected_coords[:, 0], selected_coords[:, 1]] = True

        Y_train_boot = Y_obs * O_train_boot
        p_temp = np.sum(O_train_boot) / O_train_boot.size

        U, V, _, _, _ = rpca_gd(Y_train_boot, r, sparsity, params)
        X_hat = U @ V.T
        return X_hat

    seeds = np.random.randint(0, 1e6, size=n_bootstrap)

    # Parallel execution of bootstrap iterations
    results = Parallel(n_jobs=8, verbose=1)(
        delayed(bootstrap_iteration)(seed) for seed in seeds
    )

    X_bootstrap_array = np.stack(results, axis=0)
    X_std = np.std(X_bootstrap_array, axis=0, ddof=1)

    U, V, _, _, _ = rpca_gd(Y_train, r, sparsity, params)
    X_hat = U @ V.T

    gamma = 1.3
    S = Tproj_partial((Y_obs - X_hat) * supp, gamma * sparsity * p,
                      gamma * sparsity * p)

    mask = np.logical_and(O_train, (S == 0))
    count = np.sum(mask)

    sigma = np.sqrt(np.sum(((X_hat - Y_obs) * mask) ** 2) / count)
    s_hat = np.sqrt(X_std ** 2 + sigma ** 2)

    score = np.abs(Y - X_hat) / (s_hat + 1e-8)

    observation_matrix = (S[O_cal] == 0).astype(bool)
    m = np.sum(observation_matrix)


    # Compute weights H_hat
    H_hat = np.ones((d1, d2))
    w_max = np.max((1 - supp) * H_hat)
    ww = np.zeros(int(m) + 1)
    ww[:int(m)] = H_hat[O_cal][observation_matrix].ravel()
    ww[int(m)] = w_max

    r_new = 100000
    r_all = np.append(score[O_cal][observation_matrix], r_new)
    qvals = weighted_quantile(r_all, prob=1 - alpha, w=ww)


    lo_mat = X_hat - qvals * s_hat
    up_mat = X_hat + qvals * s_hat

    return lo_mat, up_mat

def main():
    image_files = sorted(glob.glob(os.path.join("../data", "in*.jpg")))
    if len(image_files) == 0:
        print("No images found in the 'input' folder. Please check the path and filenames!")
        return
    image_files = image_files[500:700]

    sample_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    if sample_img is None:
        print("Failed to read image!")
        return
    img_shape = sample_img.shape
    n_pixels = sample_img.size
    n_frames = len(image_files)
    print("Total {} frames read, each with dimensions {}x{}".format(n_frames, img_shape[0], img_shape[1]))

    Y = np.zeros((n_frames, n_pixels), dtype=np.float32)
    for i, file in enumerate(image_files):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Error reading image:", file)
            continue
        Y[i, :] = img.flatten()

    # Normalize to [-1, 1]
    Y = (Y / 255.0) * 2 - 1

    U, S, Vh = svds(Y, k=10, which='LM')  # S is the singular value array
    S_sorted = np.sort(S)[::-1]  # Sort in descending order

    # noise = np.random.normal(0, 0.1, Y.shape)
    # Y_noisy = Y

    r = 2
    d1, d2 = Y.shape

    Omega = np.random.rand(d1, d2) < 0.8
    supp = Omega.astype(float)

    alpha = 0.1
    gamma = 1
    sparsity = 0.1
    q = 0.85
    alpha_bound = gamma * alpha
    params = {
        'step_const': 0.5,
        'max_iter': 50,
        'tol': 1e-4,
        'do_project': False
    }

    print("Starting Robust PCA decomposition...")
    t_start = time.time()
    U, V, S, _, _ = rpca_gd(Y, r, sparsity, params)
    M = U @ V.T
    lo, up = cp_rpca(Y, supp, r, sparsity, params, alpha, q)
    elapsed = time.time() - t_start
    print("Robust PCA decomposition completed in {:.4f} seconds".format(elapsed))

    frame_idx = 139
    original = Y[frame_idx, :].reshape(img_shape)
    # original_noisy = Y_noisy[frame_idx, :].reshape(img_shape)
    background1 = M[frame_idx, :].reshape(img_shape)
    foreground1 = S[frame_idx, :].reshape(img_shape)
    background2 = lo[frame_idx, :].reshape(img_shape)
    background3 = up[frame_idx, :].reshape(img_shape)
    background_disp2 = np.clip(background2, -1, 1)
    background_disp3 = np.clip(background3, -1, 1)

    # Normalize foreground image by taking absolute value
    foreground_disp1 = np.clip(np.abs(foreground1), 0, 1)
    background_disp1 = np.clip(background1, -1, 1)

    import matplotlib.gridspec as gridspec

    plt.figure(figsize=(14, 6), constrained_layout=True)  # Use constrained_layout for automatic layout adjustment
    gs = gridspec.GridSpec(2, 3, width_ratios=[0.8, 0.8, 0.8], height_ratios=[1, 1])  # Uniform width and height ratios

    # Original image
    ax1 = plt.subplot(gs[0, 0])
    plt.imshow(original, cmap='gray', aspect='auto')  # aspect='auto' to fill the subplot
    plt.title("Original Image (Frame 140)")
    plt.axis("off")

    # Lower confidence bound
    ax4 = plt.subplot(gs[1, 0])
    plt.imshow(background_disp2, cmap='gray', aspect='auto')
    plt.title("Lower Confidence Bound")
    plt.axis("off")

    # Background image
    ax2 = plt.subplot(gs[0, 1])
    plt.imshow(background_disp1, cmap='gray', aspect='auto')
    plt.title("Background Image")
    plt.axis("off")

    # Foreground image
    ax3 = plt.subplot(gs[0, 2])
    plt.imshow(foreground_disp1, cmap='gray', aspect='auto')
    plt.title("Foreground Image")
    plt.axis("off")

    # Upper confidence bound
    ax5 = plt.subplot(gs[1, 1])
    plt.imshow(background_disp3, cmap='gray', aspect='auto')
    plt.title("Upper Confidence Bound")
    plt.axis("off")

    np.random.seed(15)
    unobserved_indices = np.where(supp[frame_idx, :] == 0)[0]
    if len(unobserved_indices) < 50:
        print(f"Warning: Number of unobserved points ({len(unobserved_indices)}) is less than 50, cannot sample")
    else:
        indices = np.random.choice(unobserved_indices, 50, replace=False)
        true_values = M[frame_idx, indices]
        lo_values = lo[frame_idx, indices]
        up_values = up[frame_idx, indices]
        sort_idx = np.argsort(true_values)
        true_values_sorted = true_values[sort_idx]
        lo_values_sorted = lo_values[sort_idx]
        up_values_sorted = up_values[sort_idx]

        in_range = (lo_values_sorted <= true_values_sorted) & (true_values_sorted <= up_values_sorted)
        coverage = np.sum(in_range) / 50 * 100
        print(f"Coverage: {coverage:.2f}%")

        ax6 = plt.subplot(gs[1, 2])
        x = np.arange(1, 51)
        plt.plot(x, true_values_sorted, 'g+', label='True Values')
        plt.plot(x, lo_values_sorted, 'r-', label='Confidence Bounds')
        plt.plot(x, up_values_sorted, 'r-')
        plt.xlabel('Index', fontsize=9)
        plt.ylabel('Value', fontsize=9)
        plt.title('Confidence Intervals for 50 Random Points', fontsize=10)
        plt.legend(fontsize=8, loc='upper left')

    save_dir = "../images"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "Video background modeling.svg"), format='svg', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()