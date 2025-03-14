from docplex.mp.model import Model
from matplotlib import pyplot as plt
from rustworkx import NoEdgeBetweenNodes
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
from load_data import load_graph_from_csv
#from mystic.solvers import fmin, fmin_powell
import numpy as np
from MaxCutProblem import MaxCutProblem
from qiskit_optimization.translators import from_docplex_mp, to_ising
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit.primitives import Sampler, BackendSamplerV2, BackendSampler #samplre is deprecated, but need it to run. Why?
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
)

#TODO: move the test cases here into a more logical place in the code
import time
#import mystic
#import cvxpy as cp

class Solver():
    """
    Class which contains the ordinary cplex solver.
    Used for getting solutions to compare with quantum solution.
    TODO: Add support for max k-cut
    """
    
    def __init__(self, graph, relaxed = False, restrictions=False, k=2, vertexcover = False, verbose = False, lagrangian = 2):

        """
        Initializes the model with the given problem, but does not solve.
        If relaxed is set to true it changes the behaviour when solve is called. it will then find a local optima, 
        as cplex cannot handle non-convex continous variables.
        """
        self.vertexcover =vertexcover
        self.verbose = verbose
        if vertexcover:
            self.graph = graph
            self.model = Model(name="VertexCover")
            self.relaxed = relaxed
            if relaxed: 
                raise NotImplementedError('relaxed for vertexcover not implemented.')
                self.variables = self.model.continuous_var_list(var_multiplier*len(self.graph),lb=0,ub=1, name='x')
                self.model.parameters.optimalitytarget =2 #local minima
            else:
                self.variables = self.model.binary_var_list(len(self.graph), name='x')

            objective = 0

            #for edge in graph.edges:
            self.B = 1
            self.A = lagrangian
            for var in self.variables:
                objective += self.B*var


            for (i,j) in self.graph.edge_list(): 
                objective += self.A*(1- self.variables[i])*( 1-self.variables[j]) #negative to have max problem to align with max cut

            print('objective:', objective)
            self.objective = objective
            self.model.objective=objective
            self.model.minimize(self.objective)
        else:

            self.graph = graph
            self.model = Model(name="MaxCut")
            if k> 2:
                self.model = Model(name="Max-K-Cut")
            self.relaxed = relaxed

            var_multiplier = 1 if k==2 else k

            if relaxed: 
                self.variables = self.model.continuous_var_list(var_multiplier*len(self.graph),lb=0,ub=1, name='x')
                self.model.parameters.optimalitytarget =2 #local minima
            else:
                self.variables = self.model.binary_var_list(var_multiplier*len(self.graph), name='x')
            
            objective = 0

            #for edge in graph.edges:
    

            for (i,j, w) in self.graph.weighted_edge_list(): #TODO: extend to k-cut            
                objective+= w*(self.variables[i] + self.variables[j] - 2*self.variables[i]*self.variables[j]) 
                # w is a numpy array ( becasue i use np to generate)

            if restrictions:
                for i in range(2,len(graph), 3): #adds that every ordered tuple of three qubits most have 1 positive - for testing now, partitioning later
                    self.model.add_constraint(self.variables[i-2] + self.variables[i-1] + self.variables[i] == 1)#TODO: extend to k-cut

            self.objective = objective
            self.model.objective=objective
            self.model.maximize(self.objective)
            
    def evaluate_bitstring(self, bitstring):
        """
        Evaluates the objective value for a given bitstring.
        """
        if self.vertexcover:
            is_infeasible = 0
            for (i, j) in self.graph.edge_list():
                is_infeasible += self.A*(1 - bitstring[i]) * (1 - bitstring[j])
            if is_infeasible: return (is_infeasible + self.B*np.sum(bitstring))
            else: return self.B* np.sum(bitstring)
        objective_value = 0
        for (i, j, w) in self.graph.weighted_edge_list():
            objective_value += w * (bitstring[i] + bitstring[j] - 2 * bitstring[i] * bitstring[j])
        return objective_value
    def get_qp(self):
        return from_docplex_mp(self.model)

        

    def solve(self):
        """
        Solves the problem based on parameter relaxed.
        Returns bitstring, solution_value
        """
        if self.vertexcover:
            if self.verbose:
                print(f'Objective to minimize: {self.objective} for relaxed = {self.relaxed}')
            self.model.minimize(self.objective)
            
            solution = self.model.solve()
            print('optimal value found:',solution.get_objective_value() )
            bitstring = [var.solution_value for var in self.variables]
            if self.verbose:
                print(solution.get_objective_value(), bitstring)
            return bitstring, solution.get_objective_value()

        if self.relaxed:

                        # Define the weight matrix (from QUBO or adjacency matrix of graph)
      
            W = np.zeros((len(self.graph), len(self.graph)))
            for (i, j, w) in self.graph.weighted_edge_list():
                W[i, j] = w
                W[j, i] = w  # Assuming the graph is undirected

            n = W.shape[0]
            X = cp.Variable((n, n), PSD=True)  # PSD: Positive semidefinite
            constraints = [cp.diag(X) == 1]  # Diagonal constraints X_ii = 1

            # Objective function
            objective = cp.Maximize(cp.sum(cp.multiply(W, (1 - X))) / 4)

            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()
            # Eigendecomposition of X
            eigenvalues, eigenvectors = np.linalg.eigh(X.value)

            # Filter out negligible eigenvalues (numerical precision issues)
            valid_indices = eigenvalues > 1e-10
            eigenvalues = eigenvalues[valid_indices]
            eigenvectors = eigenvectors[:, valid_indices]

            # Form the vectors V (scaled by square root of eigenvalues)
            V = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            random_hyperplane = np.random.randn(V.shape[1])

            # Assign each vertex to a partition based on the sign of the dot product with the hyperplane
            assignments = np.sign(V @ random_hyperplane)
            
            assignments = np.where(assignments == -1, 0, assignments)
            return assignments, self.evaluate_bitstring(assignments)
                        
        if self.verbose:
            print(f'Objective to maximize: {self.objective} for relaxed = {self.relaxed}')
        self.model.maximize(self.objective)

        solution = self.model.solve()
        bitstring = [var.solution_value for var in self.variables]
        if self.verbose:
            print(solution.get_objective_value(), bitstring)
        return bitstring, solution.get_objective_value()

    
    def plot_result(self):
        """
        Plots graph of partition. Must be run after solve.
        """
        bitstring = [int(var.solution_value) for var in self.variables]


        colors = ["tab:grey" if i == 0 else "tab:purple" for i in bitstring]
        pos, default_axes = rx.spring_layout(self.graph), plt.axes(frameon=True)
        rx.visualization.mpl_draw(self.graph, node_color=colors, node_size=100, alpha=0.8, pos=pos)


def format_qaoa_samples(samples, max_len: int = 10):
    qaoa_res = []
    for s in samples:
            qaoa_res.append(("".join([str(int(_)) for _ in s.x]), s.fval, s.probability))

    res = sorted(qaoa_res, key=lambda x: -x[1])[0:max_len]

    return [(_[0] + f": value: {_[1]:.3f}, probability: {1e2*_[2]:.1f}%") for _ in res]


