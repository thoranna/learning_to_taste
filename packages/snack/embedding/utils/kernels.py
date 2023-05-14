import torch

def cauchy_kernel(distance_squared, degree:int=1):
    return degree / (degree+ distance_squared)

def euclidean_distance_squared(Y):
    sum_Y = torch.sum(Y*Y, 1)
    num = -2. * torch.mm(Y, Y.t())
    return torch.add(torch.add(num, sum_Y).t(), sum_Y)

def unscaled_student_t_prop_density(distance, distibution_dim:int=1, degree:int=3):
    return (1+ distance/degree)**(-(degree+distibution_dim)/2)

def student_t_kernel(distance_squared, degree:int=3):
    prop_density = unscaled_student_t_prop_density(distance_squared, degree=degree) 
    return prop_density

def tste_cost_function(prop_density, triplets_A, triplets_B, triplets_C):
    return prop_density[triplets_A,triplets_B] / (
        prop_density[triplets_A,triplets_B] +
        prop_density[triplets_A,triplets_C])

def t_ste_prob(x_i, x_j, x_k, alpha=1):

    nom = (1 + torch.sum((x_i - x_j)**2 / alpha, dim=1))**(-(alpha+1)/2)
    denom = (1 + torch.sum((x_i - x_j)**2 / alpha, dim=1))**(-(alpha+1)/2) + \
            (1 + torch.sum((x_i - x_k)**2 / alpha, dim=1))**(-(alpha+1)/2) + 1e-15
    return nom / denom

def distance_squared(x_i,x_j):
    return (x_i - x_j)**2

def summed_unscaled_student_t_prop_density(distance, distibution_dim:int=1, degree:int=3):
    return (1+ torch.sum(distance / degree, dim=1))**(-(degree+distibution_dim)/2)

