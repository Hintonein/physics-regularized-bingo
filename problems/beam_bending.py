import math as m
import numpy as np
import problems.data_generation_helpers as util
import torch

import scipy
from scipy import interpolate


def analytic_solution(X, k, L):
    return (1./24.)*k*X*(L**3. - 2.*L*X**2. + X**3.)


def get_pdefn(k, relative=True):
    def pdefn(X, U, bc=False):
        
        pen_mult = np.inf
        penalty = torch.ones_like(U) * pen_mult
        combined = penalty
        
        if U.grad_fn is not None:
            # store the orders of u derivatives in a list
            u_x = [U,]
            
            X_np = X[0].detach().numpy()
            
            if bc==False:
                dfn = 5
            elif bc==True:
                dfn = 3
            
            for order in range(1,dfn):
                if u_x[order-1].grad_fn is not None:
                    if order < dfn - 1:
                        u_x.append(torch.autograd.grad(u_x[order-1].sum(), X[0], create_graph=True, allow_unused=True)[0])
                    elif order == dfn-1:
                        u_x.append(torch.autograd.grad(u_x[order-1].sum(), X[0], allow_unused=True)[0])
                    if order==1:
                        pass
                    elif order==2:
                        if bc==True:
                            combined = u_x[order]
                            return [combined]
                        else:
                            pass
                    elif order==3:
                        pass
                    elif order==4:
                        combined = u_x[order] - k
                        if relative == True:
                            return [combined/k,]
                        else:
                            return [combined,]
                else:
                    return [penalty,]
        return [penalty,]

    return pdefn

def gen_training_data(k, low, high, n_b=2, n_df=4):
    
    n = 30
    X_domain = np.linspace(low,high,n)
    X_bc = np.random.choice([X_domain[0],X_domain[-1]],n_b,replace=False)[:, None]
    U_bc = analytic_solution(X_bc, k, high)
    
    X_df = np.random.choice(X_domain[1:X_domain.shape[0]-1],n_df,replace=False)[:, None]
    
    return X_bc, U_bc, X_df



