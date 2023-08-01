#
#  tsne_torch_model.py
#
# Implementation of t-SNE in pytorch. The implementation was tested on pytorch
# > 1.0, and it requires Numpy to read files. In order to plot the results,
# a working installation of matplotlib is required.
#
#
# The example can be run by executing: `python tsne_torch_model.py`
#
#
#  Created by Xiao Li on 23-03-2020.
#  Copyright (c) 2020, 2021 Xiao Li, Palle Klewitz. All rights reserved.

import sys
from typing import Union

import numpy as np
import torch
from torch import Tensor


def is_notebook():
    try:
        # noinspection PyUnresolvedReferences
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# noinspection PyPep8Naming
def _h_beta_torch(D: torch.Tensor, beta: float = 1.0):
    P = torch.exp(-D * beta)
    sumP = torch.sum(P) + 1e-10
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP
    return H, P


# noinspection PyPep8Naming
def _x2p_torch(
    X: torch.Tensor,
    tol: float = 1e-5,
    initial_dims: int = 50,
    perplexity: float = 30.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose: bool = False,
):
    """
    Performs a binary search to get P-values in such a way that each
    conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    # print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X * X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X).to(device)

    P = torch.zeros(n, n, device=device)
    beta = torch.ones(n, 1, device=device)
    logU = torch.log(torch.tensor([perplexity], device=device))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    bar = range(n)
    if verbose:
        bar = tqdm(bar)

    for i in bar:
        # Print progress
        # if i % 500 == 0:
        #     print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i] + n_list[i + 1 : n]]

        (H, thisP) = _h_beta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        h_diff = H - logU
        tries = 0
        while torch.abs(h_diff) > tol and tries < initial_dims:
            # If not, increase or decrease precision
            if h_diff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            # Recompute the values
            (H, thisP) = _h_beta_torch(Di, beta[i])

            h_diff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i] + n_list[i + 1 : n]] = thisP

    # Return final P-matrix
    return P


def _pca_torch(X, device, no_dims=50, verbose=False):
    if verbose:
        print("Preprocessing the data using PCA...")

    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    if not device == "cpu":
        with torch.cuda.device(device):
            (l, M) = torch.linalg.eig(torch.mm(X.t(), X))
    else:
        (l, M) = torch.linalg.eig(torch.mm(X.t(), X))
    # split M real
    for i in range(d):
        if l[i].imag != 0:
            M[:, i + 1] = M[:, i]
            i += 1

    M = M.real
    Y = torch.mm(X, M[:, 0:no_dims])
    return Y


# noinspection PyPep8Naming
def _tsne(
    X: Union[torch.Tensor, np.ndarray],
    no_dims: int = 2,
    initial_dims: int = 50,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    verbose: bool = False,
    device=None,
) -> Tensor:
    """
    Runs t-SNE on the dataset in the NxD array X to reduce its
    dimensionality to no_dims dimensions. The syntaxis of the function is
    `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    if not isinstance(no_dims, int) or no_dims <= 0:
        raise ValueError("dims must be positive integer")
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, device=device)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"using {device}", file=sys.stderr)

    X.to(device)

    if verbose:
        print("initializing...", file=sys.stderr)
    # Initialize variables
    if initial_dims < X.shape[1]:
        X = _pca_torch(X, device, initial_dims, verbose=verbose)
    else:
        if verbose:
            print(
                "skipping PCA because initial_dims is larger than input dimensionality",
                file=sys.stderr,
            )
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims, device=device)
    iY = torch.zeros(n, no_dims, device=device)
    gains = torch.ones(n, no_dims, device=device)

    # Compute P-values
    if verbose:
        print("computing p-values...", file=sys.stderr)
    P = _x2p_torch(X, 1e-5, initial_dims, perplexity, device=device, verbose=verbose)
    P = P + P.t()
    P = P / (torch.sum(P) + 1e-10)
    P = P * 4.0  # early exaggeration
    P = torch.max(P, torch.tensor(1e-21, device=device))  # (N, N)

    if verbose:
        print("fitting...", file=sys.stderr)

    bar = range(max_iter)

    if verbose:
        bar = tqdm(bar)

    for it in bar:
        # Compute pairwise affinities
        sum_Y = torch.sum(Y * Y, dim=1)
        num = -2.0 * torch.mm(Y, Y.t())  # (N, N)
        num = 1.0 / (1.0 + (num + sum_Y).t() + sum_Y)
        num.fill_diagonal_(0)
        Q = num / (torch.max(torch.sum(num), torch.tensor(1e-12, device=device)))
        Q = torch.max(Q, torch.tensor(1e-12, device=device))

        # Compute gradient
        PQ = P - Q
        # ((N, 1, N).repeat -> (N, no_dims, N).permute -> (N, N, no_dims)) *
        # ((N, no_dims) - (N, 1, no_dims) -> (N, N, no_dims))
        # -> (N, N, no_dims).sum[1] -> (N, no_dims)
        dY = torch.sum(
            (PQ * num).unsqueeze(1).repeat(1, no_dims, 1).transpose(2, 1)
            * (Y.unsqueeze(1) - Y),
            dim=1,
        )

        # Perform the update
        if it < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.0) != (iY > 0.0)).float() + (gains * 0.8) * (
            (dY > 0.0) == (iY > 0.0)
        ).float()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if verbose:
            C = torch.sum(P * torch.log(P / Q))
            bar.set_description(f"error: {C.cpu().item():.3f}")
            # if C.item() <= 0.8:
            #     break

        # Stop lying about P-values
        if it == 100:
            P = P / 4.0

    # Return solution
    return Y


class TorchTSNE:
    def __init__(
        self,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        n_components: int = 2,
        initial_dims: int = 50,
        verbose: bool = False,
        device=None,
    ):
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.n_components = n_components
        self.initial_dims = initial_dims
        self.verbose = verbose
        self.device = device

    # noinspection PyPep8Naming,PyUnusedLocal
    def fit_transform(self, X, y=None) -> Tensor:
        """
        Learns the t-stochastic neighbor embedding of the given data.

        :param X: ndarray or torch tensor (n_samples, *)
        :param y: ignored
        :return: ndarray (n_samples, n_components)
        """
        with torch.no_grad():
            return _tsne(
                X,
                no_dims=self.n_components,
                initial_dims=self.initial_dims,
                perplexity=self.perplexity,
                verbose=self.verbose,
                max_iter=self.n_iter,
                device=self.device,
            )


if __name__ == "__main__":
    """参数设置"""
    n_components = 2  # 降维后的特征数
    perplexity = 50  # 困惑度
    n_iter = 1000  # 迭代次数
    initial_dims = 64  # 使用PCA预先降低至此维度，以减少计算开销与显存占用
    verbose = True  # 打印信息
    device = "cuda" if torch.cuda.is_available() else "cpu"

    """随机生成测试数据"""
    n_samples = 1000
    deature_dimension = 256
    feature = torch.rand((n_samples, deature_dimension))  # 随机生成特征

    """聚类"""
    feature_emb = TorchTSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        initial_dims=initial_dims,
        verbose=verbose,
        device=device,
    ).fit_transform(feature)
