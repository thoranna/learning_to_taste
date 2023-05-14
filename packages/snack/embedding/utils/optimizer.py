import torch

class TSNEMomentum():
    def __init__(self, initial_momentum=0.5,final_momentum=0.8, momentum_switch_iter=20 ) -> None:
        self.final_momentum = final_momentum
        self.momentum = initial_momentum
        self.momentum_switch_iter = momentum_switch_iter # 250

        self.min_gain = 0.01

    def allocate_resources(self, n, no_dims):
        self.gains = torch.ones(n, no_dims)

    def check_and_set_momentum_constant(self, iter):
        # Perform the update
        if iter > self.momentum_switch_iter:
            self.momentum = self.final_momentum

    def momentum_hack_to_quickly_stop_points_going_in_the_wrong_direction(self, dY, iY):
        selection_mask = ((dY > 0.) != (iY > 0.))
        self.gains = (self.gains + 0.2) * (selection_mask).double() + (self.gains * 0.8) * (not selection_mask).double()
        self.gains[self.gains < self.min_gain] = self.min_gain


class SGDOptimizer():
    def __init__(self, learning_rate, momemtum_object:TSNEMomentum=TSNEMomentum()) -> None:
        self.learning_rate = learning_rate
        self.momentum = momemtum_object
        self.has_momemtum = momemtum_object != None

        if self.has_momemtum:
            self.take_step = lambda grad, iter: self.take_step_momemtum_gains(grad, iter)
        else:
            self.take_step = lambda grad, iter: self.take_step_gains(grad, iter)
        
    def allocate_resources(self, n, no_dims):
        self.velocity = torch.zeros(n, no_dims)
        if self.has_momemtum:
            self.momentum.allocate_resources(n, no_dims)
    
    def take_step_momemtum_gains(self, gradient, iter):
        self.momentum.check_and_set_momentum_constant(iter)
        self.velocity = self.momentum.momentum * self.velocity - self.learning_rate * (self.momentum.gains * gradient)

    def take_step_gains(self, gradient, iter):
        self.momentum.check_and_set_momentum_constant(iter)
        self.velocity  -self.learning_rate * (self.momentum.gains * gradient)