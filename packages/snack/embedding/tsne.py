import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from torch.autograd import grad
from .utils.optimizer import SGDOptimizer
from .utils.kernels import cauchy_kernel, euclidean_distance_squared
from .utils.dimensionality_reduction import pca_torch
from .utils.binary_search import x2p_torch

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)


def set_distance_to_own_point_to_zero(distance_matrix, n):
    distance_matrix[range(n), range(n)] = 0.

def prevent_values_being_too_low(Q, tolerance=torch.tensor([1e-12])):
    return torch.max(Q, tolerance)


class TSNEGrad():
    def __init__(self, no_dims) -> None:
        self.no_dims = no_dims

    def allocate_resources(self, n):
        self.dY = torch.zeros(n, self.no_dims)

    def force_acting_on_spring(self, PQ, cauchy_numerator, i:int):
        return (PQ[:, i] * cauchy_numerator[:, i]).repeat(self.no_dims, 1).t()

    def spring_between_yi_yj(self, Y, i):
        return (Y[i, :] - Y)

    def force_exterted_on_yi(self, PQ, cauchy_numerator, Y, i):
        return torch.sum(self.force_acting_on_spring( PQ, cauchy_numerator, i)* self.spring_between_yi_yj(Y, i),0)

    def tsne_grad(self, PQ, cauchy_numerator, Y, n):
        for i in range(n):
            self.dY[i, :] = self.force_exterted_on_yi(PQ, cauchy_numerator, Y, i)


class TSNE():
    def __init__(self, no_dims:int=2, perplexity=30.0, optimizer:SGDOptimizer = SGDOptimizer(500) ) -> None:
        self.max_iter = 1000
        self.initial_dims = 50
        self.no_dims = no_dims
        self.perplexity = perplexity
        self.grad_calculator = TSNEGrad(no_dims)
        self.optim = optimizer
        self.crit = torch.nn.KLDivLoss(reduction="sum")

    def allocate_resources(self, n):
        self.Y =  torch.Tensor(torch.rand(n, self.no_dims, dtype=torch.float))
        self.Y.requires_grad=True
        self.dY = torch.zeros(n, self.no_dims)
        self.gains = torch.ones(n, self.no_dims)

    def initialize_distance_using_pca(self, X):
        return pca_torch(X, self.initial_dims)

    def calculate_joint_probabilities_from_distances(self, distances):
        conditional_P = x2p_torch(distances, 1e-5, self.perplexity)
        P = conditional_P + conditional_P.t()
        P_ij = P / torch.sum(P)
        return prevent_values_being_too_low(P_ij, torch.tensor([1e-21]))

    def cauchy_kernel_numerator(self, Y, n):
        distance = euclidean_distance_squared(Y)
        distance_kernelized = cauchy_kernel(distance)
        set_distance_to_own_point_to_zero(distance_kernelized, n)
        return distance_kernelized

    def calculate_lower_dimensional_joint_probabilities(self, cauchy_numerator):
        lower_dim_similarities_q_ij = cauchy_numerator / torch.sum(cauchy_numerator)
        return  prevent_values_being_too_low(lower_dim_similarities_q_ij) 

    def tsne_embed(self, X):
        distances = self.initialize_distance_using_pca(X)
        (n, d) = distances.shape

        self.allocate_resources(n)
        self.grad_calculator.allocate_resources(n)
        self.optim.allocate_resources(n, self.no_dims)

        optimizer = torch.optim.SGD(params=[self.Y], lr=1, momentum=0.8)

        P_ij = self.calculate_joint_probabilities_from_distances(distances)
        P_ij = P_ij * 4.    # early exaggeration

        for iter in range(self.max_iter):
            distance_kernelized = self.cauchy_kernel_numerator(self.Y, n)
            Q_ij = self.calculate_lower_dimensional_joint_probabilities(distance_kernelized)

            batch_loss = torch.sum(P_ij * torch.log(P_ij / Q_ij)) #self.crit(Q_ij.log(),P_ij) #
            
            # optimizer.zero_grad()
            # batch_loss.backward()
            # optimizer.step()

            PQ = P_ij - Q_ij

            self.grad_calculator.tsne_grad(PQ, distance_kernelized, self.Y, n)

            self.optim.take_step(4*self.grad_calculator.dY,iter)
            gradient = grad(batch_loss, self.Y)[0]

            self.Y = self.Y + self.optim.velocity
            self.Y = self.Y - torch.mean(self.Y, 0)

            if (iter + 1) % 10 == 0:
                C = torch.sum(P_ij * torch.log(P_ij / Q_ij))

            # Stop lying about P-values
            if iter == 100:
                P_ij = P_ij / 4.

        # Return solution
        return self.Y



if __name__ == "__main__":
    X = np.loadtxt("mnist2500_X.txt")
    X -= np.mean(X, 0)
    X /= np.max(np.abs(X))
    X = torch.Tensor(X)
    labels = np.array(np.loadtxt("mnist2500_labels.txt").tolist(), dtype=int)

    # confirm that x file get same number point than label file
    # otherwise may cause error in scatter
    assert(len(X[:, 0])==len(X[:,1]))
    assert(len(X)==len(labels))


    tsne_obj = TSNE()
    tsne_obj.tsne_embed(X)
    Y = tsne_obj.Y

    if True:
        Y = Y.cpu().detach().numpy()

    jet = plt.cm.jet
    colors = jet(np.linspace(0.2, 1, len(set(labels))))    
    unique_labels = list(set(labels))
    unique_labels.sort()

    plt.figure()
    labels = np.array(labels)
    for color, unique_label in zip(colors,unique_labels):
        plt.plot(Y[labels==unique_label,0], Y[labels==unique_label, 1],  "o", label="class_"+str(unique_label), color=color)
    plt.legend()
    plt.show()
