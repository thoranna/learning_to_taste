
import torch

def pca_torch(X, no_dims=50):
    (n, d) = X.shape
    X = X - torch.mean(X, 0)
    (l, M) = torch.linalg.eig(torch.mm(X.t(), X))

    # split M real
    # this part may be some difference for complex eigenvalue
    # but complex eignevalue is meanless here, so they are replaced by their real part
    i = 0
    while i < d:
        if l[i].imag != 0:
            M[:, i+1] = M[:, i]
            i += 2
        else:
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims].real)
    return Y