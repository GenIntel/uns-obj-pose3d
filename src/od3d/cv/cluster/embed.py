import torch
import numpy as np
from sklearn.manifold import TSNE

def pca(X, C=2, center=True):
    """
    Principal Component Analysis (PCA) is a linear dimensionality reduction
    Args:
        X (torch.Tensor): NxF

    Returns:
        X_embedded (torch.Tensor): NxC
    """

    # if q==None: q=min(6,N)
    _, _, pca_V = torch.pca_lowrank(X, center=center, q=C)
    X_embedded = torch.mm(X, pca_V[:, :C])
    return X_embedded

def tsne(X, C=2):
    """
    t-distributed Stochastic Neighbor Embedding (t-SNE) is a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions. Specifically, it models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points with high probability.
    Args:
        X (torch.Tensor): NxF
    Returns:
        X_embedded (torch.Tensor): NxC
    """
    device = X.device

    X_embedded = torch.from_numpy(TSNE(n_components=C).fit_transform(X.detach().cpu().numpy())).to(device=device)
    return X_embedded

