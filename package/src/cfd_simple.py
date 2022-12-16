import torch

class simple():
    def __init__(self, domain_rows, domain_cols, simulation_timestep):
        self.rows = domain_rows
        self.cols = domain_cols
        self.u = torch.zeros(domain_rows+1, domain_cols)
        self.v = torch.zeros(domain_rows, domain_cols+1)
        self.p = torch.zeros(domain_rows, domain_cols)
        self.ts = simulation_timestep

    def momentum_gradient(self):
        