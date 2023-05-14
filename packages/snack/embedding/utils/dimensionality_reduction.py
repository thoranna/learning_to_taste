
import torch

def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    print("this is n: ", n)
    print("this is d: ", d)
    X = X - torch.mean(X, 0)
    print("this is X: ", X)

    (l, M) = torch.linalg.eig(torch.mm(X.t(), X))

    print("this is l: ", l)
    print("this is M: ", M)

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