import torch

import numpy as np

from bingo.evaluation.fitness_function import VectorBasedFunction


class DifferentialRegression_TF(VectorBasedFunction):

    def __init__(self,
                 X, U, X_df, df_err,
                 df_order=1,
                 neumann_bc=False,
                 differential_weight=1.0,
                 metric="rmse",
                 clo_type = 'optimize',
                 detect_const_solutions=True):

        super().__init__(None, metric)
        
        self.X = [torch.tensor(X[:, i], dtype=torch.float64)
                  for i in range(X.shape[1])]
        
        for X in self.X:
            X.requires_grad = True
        
        self.U = U  

        self.X_df = [torch.tensor(X_df[:, i], dtype=torch.float64)
                     for i in range(X_df.shape[1])]
        
        for X in self.X_df:
            X.requires_grad = True
       
        self.differential_weight = differential_weight
        self.detect_const_solutions = detect_const_solutions

        self.df_err = df_err
        self.clo_type = clo_type
        self.neumann_bc = neumann_bc
        

    def build_torch_graph_from_agraph(self, individual):
        
        commands = individual._simplified_command_array
        constants = individual._simplified_constants
        
        def evaluate(X):
            ad_stack = [None] * commands.shape[0]

            for i in range(commands.shape[0]):

                node = commands[i, 0]
                if node == 0:
                    column_idx = commands[i, 1]
                    ad_stack[i] = X[column_idx]
                elif node == 1:
                    const_idx = commands[i, 1]
                    ad_stack[i] = torch.ones_like(
                        X[0], dtype=torch.float64) * constants[const_idx]
                elif node == 2:
                    t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                    ad_stack[i] = ad_stack[t1_idx] + ad_stack[t2_idx]
                elif node == 3:
                    t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                    ad_stack[i] = ad_stack[t1_idx] - ad_stack[t2_idx]
                elif node == 4:
                    t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                    ad_stack[i] = ad_stack[t1_idx] * ad_stack[t2_idx]
                elif node == 5:
                    t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                    ad_stack[i] = ad_stack[t1_idx] / ad_stack[t2_idx]
                elif node == 6:
                    t1_idx = commands[i, 1]
                    ad_stack[i] = torch.sin(ad_stack[t1_idx])
                elif node == 7:
                    t1_idx = commands[i, 1]
                    ad_stack[i] = torch.cos(ad_stack[t1_idx])
                elif node == 8:
                    t1_idx = commands[i, 1]
                    ad_stack[i] = torch.exp(ad_stack[t1_idx])
                elif node == 9:
                    t1_idx = commands[i, 1]
                    ad_stack[i] = torch.log(torch.abs(ad_stack[t1_idx]))
                elif node == 10:
                    t1_idx, t2_idx = commands[i, 1], commands[i, 2]
                    ad_stack[i] = torch.pow(
                        torch.abs(ad_stack[t1_idx]), ad_stack[t2_idx])
                elif node == 11:
                    t1_idx = commands[i, 1]
                    ad_stack[i] = torch.abs(ad_stack[t1_idx])
                elif node == 12:
                    t1_idx = commands[i, 1]
                    ad_stack[i] = torch.sqrt(torch.abs(ad_stack[t1_idx]))
                else:
                    raise IndexError(f"Node value {node} unrecognized")

            return ad_stack[-1]

        return evaluate
    
    
    def evaluate_fitness_vector(self, individual):

        self.eval_count += 1
        
        ad_graph_function = self.build_torch_graph_from_agraph(individual)
        
        U_hat = ad_graph_function(self.X)
        
        error_fit = self.U[:, 0] - U_hat.detach().numpy()
        
        if self.clo_type == 'optimize':
            fitness = self._metric(error_fit)
        elif self.clo_type == 'root':
            fitness = error_fit
        
        
        if self.neumann_bc == True:
            
            errors_neumann_bc = self.df_err(self.X, U_hat, bc=True)
            for error_neumann_bc in errors_neumann_bc:
                error_neumann_bc = error_neumann_bc.detach().numpy()
                if self.clo_type == 'optimize':
                    fitness +=  self._metric(error_neumann_bc)
                elif self.clo_type == 'root':
                    fitness  = np.concatenate((fitness, error_neumann_bc), axis=0)
        
        U_df = ad_graph_function(self.X_df)
        
        errors_df = self.df_err(self.X_df, U_df)
        
        for i,error_df in enumerate(errors_df):
            error_df = error_df.detach().numpy()
            if self.clo_type == 'optimize':
                fitness += self.differential_weight * self._metric(error_df)
            elif self.clo_type == 'root':
                fitness  = np.concatenate((fitness, error_df), axis=0)
                
        return fitness
    