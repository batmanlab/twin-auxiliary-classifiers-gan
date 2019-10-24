from __future__ import division, print_function

import os.path, sys, tarfile
import numpy as np
from scipy import linalg
from six.moves import range, urllib
from sklearn.metrics.pairwise import polynomial_kernel
from tqdm import tqdm
import torch


# from tqdm docs: https://pypi.python.org/pypi/tqdm#hooks-and-callbacks
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # also sets self.n = b * bsize





def get_splits(n, splits=10, split_method='openai'):
    if split_method == 'openai':
        return [slice(i * n // splits, (i + 1) * n // splits)
                for i in range(splits)]
    elif split_method == 'bootstrap':
        return [np.random.choice(n, n) for _ in range(splits)]
    else:
        raise ValueError("bad split_method {}".format(split_method))


def inception_score(preds, **split_args):
    split_inds = get_splits(preds.shape[0], **split_args)
    scores = np.zeros(len(split_inds))
    for i, inds in enumerate(split_inds):
        part = preds[inds]
        kl = part * (np.log(part) - np.log(np.mean(part, 0, keepdims=True)))
        kl = np.mean(np.sum(kl, 1))
        scores[i] = np.exp(kl)
    return scores


def fid_score(codes_g, codes_r, eps=1e-6, output=sys.stdout, **split_args):
    splits_g = get_splits(codes_g.shape[0], **split_args)
    splits_r = get_splits(codes_r.shape[0], **split_args)
    assert len(splits_g) == len(splits_r)
    d = codes_g.shape[1]
    assert codes_r.shape[1] == d

    scores = np.zeros(len(splits_g))
    with tqdm(splits_g, desc='FID', file=output) as bar:
        for i, (w_g, w_r) in enumerate(zip(bar, splits_r)):
            part_g = codes_g[w_g]
            part_r = codes_r[w_r]

            mn_g = part_g.mean(axis=0)
            mn_r = part_r.mean(axis=0)

            cov_g = np.cov(part_g, rowvar=False)
            cov_r = np.cov(part_r, rowvar=False)

            covmean, _ = linalg.sqrtm(cov_g.dot(cov_r), disp=False)
            if not np.isfinite(covmean).all():
                cov_g[range(d), range(d)] += eps
                cov_r[range(d), range(d)] += eps
                covmean = linalg.sqrtm(cov_g.dot(cov_r))

            scores[i] = np.sum((mn_g - mn_r) ** 2) + (
                np.trace(cov_g) + np.trace(cov_r) - 2 * np.trace(covmean))
            bar.set_postfix({'mean': scores[:i+1].mean()})
    return scores


def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                            ret_var=True, output=sys.stdout, **kernel_args):
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
        for i in bar:
            g = codes_g[choice(len(codes_g), subset_size, replace=False)]
            r = codes_r[choice(len(codes_r), subset_size, replace=False)]
            o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
            if ret_var:
                mmds[i], vars[i] = o
            else:
                mmds[i] = o
            bar.set_postfix({'mean': mmds[:i+1].mean()})
    return (mmds, vars) if ret_var else mmds


def polynomial_mmd(codes_g, codes_r, degree=2, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True,sample=10000):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    sample_g = np.random.choice(codes_g.shape[0],sample)
    sample_r = np.random.choice(codes_r.shape[0], sample)
    X = codes_g[sample_g]
    Y = codes_r[sample_r]

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY,
                              var_at_m=var_at_m, ret_var=ret_var)


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est

# data1 = torch.randn(12800)
# data2 = torch.randn(12800)*2 + 3
# data3 = torch.randn(12800)*3 + 4
#
# data = torch.cat([data1,data2,data3],dim=0).unsqueeze(dim=1).numpy()
#
# print(polynomial_mmd(data,data*2))