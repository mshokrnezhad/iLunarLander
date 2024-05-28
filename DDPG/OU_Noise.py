#       In this file, we implement a class for noise gneration based on the Ornstein–Uhlenbeck process.

import numpy as np

class OU_Noise:
    def __init__(self, mu, sigma = 0.5, theta = 0.2, dt = 1e-2, x0 = None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        
        self.reset()

    def __call__(self):
        dif_ = self.mu - self.x_prev
        rnd_ = np.random.normal(size = self.mu.shape)
        x = self.x_prev + self.theta * self.dt * dif_ + self.sigma + np.sqrt(self.dt) + rnd_
        
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)    