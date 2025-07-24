import numpy as np
import scipy.sparse as sp
import scipy.linalg as scilinalg
from SPG import *
from proj import *
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd

def Ai(X, supp):
    return X.multiply(supp) if sp.issparse(X) else X * supp

def calc_para(M_star, r):
    # Calculate the incoherence parameter, maximum singular value, and condition number.
    U, Sigma, Vt = svds(M_star.astype(float), k=r, which='LM')
    Sigma = np.sort(Sigma)[::-1]
    a1 = np.max(np.sum(U ** 2, axis=1))
    a2 = np.max(np.sum(Vt.T ** 2, axis=1))
    mu = max(a1 * M_star.shape[0] / r, a2 * M_star.shape[1] / r)
    sigma1 = Sigma[0]
    cond_num = sigma1 / Sigma[-1] if Sigma[-1] > 1e-10 else sigma1 / 1e-10
    return mu, sigma1, cond_num

def Tproj_partial(S, a_col, a_row):
    # Truncated threshold projection: Retain the largest absolute value element in each column/row.
    d1, d2 = S.shape
    kcol = int(a_col * d1)
    krow = int(a_row * d2)
    col_idx = np.argpartition(np.abs(S), -kcol, axis=0)[-kcol:]
    mask_col = np.zeros_like(S)
    for j in range(d2):
        mask_col[col_idx[:, j], j] = 1
    row_idx = np.argpartition(np.abs(S.T), -krow, axis=0)[-krow:]
    mask_row = np.zeros_like(S)
    for i in range(d1):
        mask_row[i, row_idx[:, i]] = 1
    return S * mask_col * mask_row

def HT(S0, k):
    # Perform hard thresholding on matrix S0, retaining the k elements with the largest absolute values.
    S_flat = S0.flatten()
    if k == 0:
        return np.zeros_like(S0)
    threshold = np.sort(np.abs(S_flat))[-k]
    mask = np.abs(S0) >= threshold
    return S0 * mask

def soft_threshold(X, tau):
    """Soft thresholding operator for L1 norm (for sparse matrix S)."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def svd_threshold(X, tau):
    """Singular value thresholding for nuclear norm (for low-rank matrix L)."""
    # Handle inf/NaN and clip extreme values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    X = np.clip(X, -1e10, 1e10)  # Clip extreme values
    try:
        # Use randomized SVD for better stability
        U, s, Vt = randomized_svd(X, n_components=min(X.shape) - 1, random_state=0)
        s = np.maximum(s - tau, 0)
        return U @ np.diag(s) @ Vt
    except Exception as e:
        print(f"SVD failed: {e}")
        return X  # Fallback to input matrix

def f_(x, q):
    # The zoomed logical function
    return q / (1 + np.exp(-x))

def fprime(x, q):
    return q * np.exp(x) / (1 + np.exp(x)) ** 2

def logObjectiveGeneral(x, y, idx, f, fprime):
    # Calculate the negative log-likelihood loss and its gradient.
    F = -np.sum(np.log(y[idx] * f(x[idx]) - (y[idx] - 1) / 2))
    G = np.zeros(len(x))
    v = (y[idx] * f(x[idx]) - (y[idx] - 1) / 2)
    w = -fprime(x[idx])
    G[idx] = w / v
    return F, G

def projectNuclear(B, d1, d2, radius, alpha):
    # Project matrix B onto the nuclear norm ball.
    U, S, Vh = np.linalg.svd(B.reshape((d1, d2)), full_matrices=False)
    s2 = euclidean_proj_l1ball(S, radius)
    B_proj = U @ np.diag(s2) @ Vh
    return B_proj.reshape((d1 * d2,))

def estimate_P(Y_train, q, P, missing_model='homo'):
    # Estimated observation probability matrix
    d1, d2 = Y_train.shape
    if missing_model == "oracle":
        P_hat = P
    elif missing_model == "homo":
        P_hat = (1/q) * np.mean(Y_train) * np.ones((d1, d2))
    elif missing_model == 'logis1' or missing_model == 'logis2':
        yy = 2*(Y_train-0.5).ravel()
        x_init = np.zeros(d1*d2)
        idx = range(d1*d2)
        const = 1.0
        if missing_model == 'logis1':
            k_l = 5
        else:
            k_l = 1
        radius = const * np.sqrt(d1*d2*k_l)
        f_loc = lambda x: f_(x, q)
        fprime_loc = lambda x: fprime(x, q)
        funObj = lambda x_var: logObjectiveGeneral(x_var, yy, idx, f_loc, fprime_loc)
        funProj = lambda x_var: projectNuclear(x_var, d1, d2, radius, const)
        default_options = SPGOptions()
        default_options.maxIter = 10000
        default_options.verbose = 2
        default_options.suffDec = 1e-4
        default_options.progTol = 1e-9
        default_options.optTol = 1e-9
        default_options.curvilinear = 1
        default_options.memory = 10
        default_options.useSpectral = True
        default_options.bbType = 1
        default_options.interp = 2
        default_options.numdiff = 0
        default_options.testOpt = True
        spg_options = default_options
        x_, F_ = SPG(funObj, funProj, x_init, spg_options)
        A_hat = x_.reshape((d1, d2))
        U, s_hat, Vh = np.linalg.svd(A_hat)
        M_d = U[:, :k_l] @ np.diag(s_hat[:k_l]) @ Vh[:k_l, :]
        P_hat = (1/q) * f_(M_d, q).reshape((d1, d2))
    elif missing_model=='rank1':
        u_hat, s_hat, vt_hat = scilinalg.svd(Y_train, full_matrices=False)
        u_hat, s_hat, vt_hat = u_hat[:, :1], s_hat[:1], vt_hat[:1, :]
        P_hat = (1/q) * u_hat @ np.diag(s_hat) @ vt_hat
    return P_hat

def weighted_quantile(v, prob, w):
    # Calculate the weighted quantile
    if len(w) == 0:
        w = np.ones(len(v))
    o = np.argsort(v)
    v = v[o]
    w = w[o]
    i = np.where(np.cumsum(w/np.sum(w)) >= prob)
    if len(i[0]) == 0:
        return float('inf')
    else:
        return v[np.min(i)]

def compute_Sigma_gaussian(Mhat, r, p_observe, sigma_est):
    # Calculate the noise variance matrix sigmaS
    u, s, vh = scilinalg.svd(Mhat, full_matrices=False)
    U = u[:, :r]
    V = vh[:r, :].T
    d1, d2 = Mhat.shape
    # U_ = np.diag(U @ U.T).reshape((d1, 1))
    # V_ = np.diag(V @ V.T).reshape((d2, 1))
    U_ = np.sum(U ** 2, axis=1).reshape((d1, 1))
    V_ = np.sum(V ** 2, axis=1).reshape((d2, 1))

    sigmaS = U_ @ np.ones((1, d2)) + np.ones((d1, 1)) @ V_.T
    sigmaS /= p_observe
    sigmaS = sigma_est * np.sqrt(sigmaS)
    return sigmaS
