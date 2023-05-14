import torch
import numpy as np
from torch.autograd import grad
import matplotlib.pyplot as plt
from .utils.optimizer import SGDOptimizer
from .tsne import TSNE
from .tste import TSTE
from ..datasets.mnist import MNIST2KDataset, TripletMNIST2K
from sklearn.neighbors import NearestNeighbors
from .utils.misc import to_torch_and_device
from ..datamodule.metrics import post_hoc_metric_given_embedding_and_triplets

# Added for reproducability
np.random.seed(42)

class SNaCK():
    def __init__(self, num_elements, no_dims:int=2, perplexity=30.0, contrib_cost_triplets=0.05, contrib_cost_tsne=10, optimizer:SGDOptimizer = SGDOptimizer(1), max_iter=1000, patience=5) -> None:
        self.tste_obj = TSTE(num_elements, no_dims, optimizer)
        self.tsne_obj = TSNE(no_dims, perplexity, optimizer)

        self.num_elements = num_elements
        self.max_iter = max_iter
        self.contrib_cost_triplets = contrib_cost_triplets
        self.contrib_cost_tsne = contrib_cost_tsne
        self.no_dims = no_dims
        self.perplexity = perplexity
        self.optim = optimizer
        self.optim.allocate_resources(num_elements, no_dims)
        self.crit = torch.nn.KLDivLoss(reduction="sum")
        self.allocate_resources(num_elements)
        #self.optimizer = torch.optim.Adam(params=[self.Y], lr=0.1, betas=(0.8,0.9))
        self.logs = []
        self.labels = None
        self.logging_turned_on = False
        self.use_early_stopping = True
        if self.use_early_stopping:
            self.patience = patience
            self.last_loss=1000000
            self.triggertimes=0
            self.should_return_embedding=False
            

    def log_metrics(self, logging_turned_on=False):
        if logging_turned_on and self.labels!= None:
            self.logging_turned_on = True
        else:
            print("Loggining not turned on. Provide labels")

    def allocate_resources(self, n):
        self.Y = torch.Tensor(torch.rand(n, self.no_dims, dtype=torch.float))
        self.Y.requires_grad=True

    def snack_embed(self, X, triplets):
        X = to_torch_and_device(X)
        self.contrib_cost_triplets = self.contrib_cost_triplets*(2.0 / float(len(triplets)) * float(self.num_elements))

        distances = self.tsne_obj.initialize_distance_using_pca(X)
        P_ij = self.tsne_obj.calculate_joint_probabilities_from_distances(distances)
        P_ij = P_ij * 4. # early exaggeration

        batch_size, batches = self.tste_obj.prepare_batches(triplets.shape[0])
        triplets = to_torch_and_device(triplets)
        
        for iter in range(self.max_iter):
            perm = np.random.permutation(batches)
            
            for batch_ind in perm:
                distance_kernelized = self.tsne_obj.cauchy_kernel_numerator(self.Y, self.num_elements)
                Q_ij = self.tsne_obj.calculate_lower_dimensional_joint_probabilities(distance_kernelized)
                tsne_loss = self.crit(Q_ij.log(),P_ij)
                tsne_gradient = grad(tsne_loss, self.Y)[0]

                batch_trips = triplets[batch_ind * batch_size: (batch_ind + 1) * batch_size, ] 
                prob = self.tste_obj.forward_step(self.Y, batch_trips)
                tste_loss = self.tste_obj.calculate_tste_loss(prob)
                tste_gradient = grad(tste_loss, self.Y)[0]

                self.optim.take_step(self.contrib_cost_triplets*tste_gradient + self.contrib_cost_tsne*tsne_gradient ,iter)
                self.Y = self.Y + self.optim.velocity

                if (iter + 1) % 10 == 0:
                    C = torch.sum(self.contrib_cost_triplets*tste_loss + self.contrib_cost_tsne*tsne_loss).detach().cpu().numpy().item()
                    print("Iteration %d: error is %f" % (iter + 1, C))
                    print("Iteration %d: grad is %f" % (iter + 1, torch.sum(self.contrib_cost_triplets*tste_gradient + self.contrib_cost_tsne*tsne_gradient).detach().cpu().numpy().item()))
                    if C> self.last_loss:
                        self.triggertimes +=1
                        print('Trigger Times:', self.triggertimes)
                        if self.triggertimes>=self.patience:
                            print('Early stopping!\nStart to test process.')
                            self.should_return_embedding = True
                            break
                    else:
                        self.triggertimes = 0

                    self.last_loss = C           

                # Stop lying about P-values
                if iter == 100:
                    P_ij = P_ij / 4.
            
            if self.should_return_embedding:
                break

            if self.logging_turned_on:
                self.logs.append(post_hoc_metric_given_embedding_and_triplets(self.Y.detach().cpu().numpy(), batch_trips.cpu().numpy(), self.labels))

        # Return solution
        return self.Y


if __name__ == "__main__":

    imgs= "mnist2500_X.txt"
    labels= "mnist2500_labels.txt"
    msnitdata = MNIST2KDataset(imgs,labels)
    labels = list(msnitdata.labels)

    triplet_train_dataset = TripletMNIST2K(msnitdata) # Returns triplets of images
    ftrs = np.array(msnitdata.imgs, dtype=np.float32)

    no_dims:int = 2
    N, D =  ftrs.shape #2500, 42 for mnsit  
    alpha:int = no_dims-1
    triplets = np.array(triplet_train_dataset.test_triplets[0:1000])

    ftrs -= np.mean(ftrs, 0)
    ftrs /= np.max(np.abs(ftrs))

    snack = SNaCK(N)
    snack.labels = labels
    snack.log_metrics(True)
    Y = snack.snack_embed(ftrs, triplets)
    Y = Y.cpu().detach().numpy()

    jet = plt.cm.jet
    colors = jet(np.linspace(0.2, 1, len(set(labels))))    
    unique_labels = list(set(labels))
    unique_labels.sort()

    fig, ax = plt.subplots(figsize=(6,6))
    labels = np.array(labels)
    for color, unique_label in zip(colors,unique_labels):
        ax.scatter(Y[labels==unique_label,0], Y[labels==unique_label, 1], lw=0, c=color, s=30, label="class_"+str(unique_label), alpha=0.5)
    ax.grid(True)
    ax.legend()
    plt.show()

    print("done")