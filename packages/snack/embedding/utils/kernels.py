import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def cauchy_kernel(distance_squared, degree:int=1):
    set_seed(42)
    return degree / (degree+ distance_squared)

def euclidean_distance_squared(Y):
    set_seed(42)
    sum_Y = torch.sum(Y*Y, 1)
    num = -2. * torch.mm(Y, Y.t())
    return torch.add(torch.add(num, sum_Y).t(), sum_Y)

def unscaled_student_t_prop_density(distance, distibution_dim:int=1, degree:int=3):
    set_seed(42)
    return (1+ distance/degree)**(-(degree+distibution_dim)/2)

def student_t_kernel(distance_squared, degree:int=3):
    set_seed(42)
    prop_density = unscaled_student_t_prop_density(distance_squared, degree=degree) 
    return prop_density

def tste_cost_function(prop_density, triplets_A, triplets_B, triplets_C):
    set_seed(42)
    return prop_density[triplets_A,triplets_B] / (
        prop_density[triplets_A,triplets_B] +
        prop_density[triplets_A,triplets_C])

def t_ste_prob(x_i, x_j, x_k, alpha=1):
    set_seed(42)
    nom = (1 + torch.sum((x_i - x_j)**2 / alpha, dim=1))**(-(alpha+1)/2)
    denom = (1 + torch.sum((x_i - x_j)**2 / alpha, dim=1))**(-(alpha+1)/2) + \
            (1 + torch.sum((x_i - x_k)**2 / alpha, dim=1))**(-(alpha+1)/2) + 1e-15
    return nom / denom

def distance_squared(x_i,x_j):
    set_seed(42)
    return (x_i - x_j)**2

def summed_unscaled_student_t_prop_density(distance, distibution_dim:int=1, degree:int=3):
    set_seed(42)
    return (1+ torch.sum(distance / degree, dim=1))**(-(degree+distibution_dim)/2)

