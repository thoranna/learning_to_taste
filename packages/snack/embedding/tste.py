import torch
import numpy as np
from torch.autograd import grad
import matplotlib.pyplot as plt
from .utils.optimizer import SGDOptimizer
from .utils.kernels import  distance_squared, summed_unscaled_student_t_prop_density
from ..datasets.mnist import MNIST2KDataset, TripletMNIST2K
from .utils.misc import to_torch_and_device
import random

if torch.cuda.is_available():
    print("set use cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

# Added for reproducability
np.random.seed(42)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class TSTE():
    def __init__(self, N, no_dims:int=2, optimizer:SGDOptimizer = SGDOptimizer(1)) -> None:
        set_seed(42)
        self.epochs = 300
        self.no_dims = no_dims
        self.optim = optimizer
        self.allocate_resources(N)
        self.optim.allocate_resources(N, self.no_dims)
        self.optimizer = torch.optim.SGD(params=[self.Y], lr=1, momentum=0.5)

    def allocate_resources(self, n):
        self.Y = torch.Tensor(torch.rand(n, self.no_dims, dtype=torch.float))
        self.Y.requires_grad=True

    def calculate_tste_probability(self, x_i, x_j, x_k, degree):
        positive_distance = distance_squared(x_i, x_j) / degree
        negative_distance = distance_squared(x_i, x_k) / degree
        nom = summed_unscaled_student_t_prop_density(positive_distance, degree=degree)
        denom = (nom + summed_unscaled_student_t_prop_density(negative_distance, degree=degree))+ 1e-15
        return nom / denom

    def calculate_tste_loss(self, tste_prob):
        return -torch.sum(torch.log(tste_prob))

    def auto_calculate_tste_gradient(self, tste_loss):
        return grad(tste_loss, self.Y)[0]

    def prepare_batches(self, n):
        batch_size = n
        batches = n // batch_size
        return batch_size, batches

    def forward_step(self, Y, batch_trips):
         # a batch of triplets
        print("Y.shape: ", Y.shape, "batch_trips: ", batch_trips)
        batch_Xs = Y[batch_trips, :]  # items involved in the triplet
        prob = self.calculate_tste_probability(batch_Xs[:, 0, :].squeeze(),
                        batch_Xs[:, 1, :].squeeze(),
                        batch_Xs[:, 2, :].squeeze(),
                        self.no_dims - 1)
        return prob

    def tste_embed(self, triplets):

        batch_size, batches = self.prepare_batches(triplets.shape[0])
        triplets = to_torch_and_device(triplets)

        for it in range(self.epochs):
            print('epoch', it)
            perm = np.random.permutation(batches)
            
            for batch_ind in perm:
                batch_trips = triplets[batch_ind * batch_size: (batch_ind + 1) * batch_size, ] 
                prob = self.forward_step(self.Y, batch_trips)

                loss = self.calculate_tste_loss(prob)

                #### same as above but calculates the distance for all triplets, i.e there is no batching!
                #distance_squared = euclidean_distance_squared(self.Y[batch_ind * batch_size: (batch_ind + 1) * batch_size, ])
                #prob = self.calculate_tste_prop(distance_squared , batch_trips[:, 0],
                #                batch_trips[:, 1],
                #                batch_trips[:, 2])
                #loss2 = self.calculate_tste_loss(prob)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #### manual gradient calculation and own SGD update
                #gradient = self.auto_calculate_tste_gradient(loss)
                #self.optim.take_step(gradient,it)
                #self.Y = self.Y + self.optim.velocity
                #self.Y = self.Y - torch.mean(self.Y, 0)

        # Return solution
        return self.Y



if __name__ == "__main__":

    imgs = "mnist2500_X.txt"
    labels = "mnist2500_labels.txt"
    msnitdata = MNIST2KDataset(imgs, labels)
    labels = list(msnitdata.labels)
    triplet_train_dataset = TripletMNIST2K(msnitdata) # Returns triplets of images

    # print("this is the triplet train dataset: ", triplet_train_dataset.test_triplets)
    # print(triplet_train_dataset.labels_set)
    # print(triplet_train_dataset.label_to_indices)

    no_dims = 2
    N, D = 2500, 42
    alpha = no_dims - 1
    triplets = np.array(triplet_train_dataset.test_triplets)

    tste_obj = TSTE(N)
    Y = tste_obj.tste_embed(triplets)
    Y = Y.cpu().detach().numpy()

    jet = plt.cm.jet
    colors = jet(np.linspace(0.2, 1, len(set(labels))))    
    unique_labels = list(set(labels))
    unique_labels.sort()

    print("these are labels: ", labels)
    print("these are unique labels: ", unique_labels)
    # plt.figure()
    # labels = np.array(labels)
    # for color, unique_label in zip(colors, unique_labels):
    #    plt.plot(Y[labels==unique_label, 0], Y[labels==unique_label, 1],  "o", label="class_" + str(unique_label), color=color)
    # plt.legend()
    # plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*Y.T, lw=0, s=1, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    plt.show()