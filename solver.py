from docplex.mp.model import Model
from matplotlib import pyplot as plt
import rustworkx as rx
import numpy as np
from qiskit_optimization.translators import from_docplex_mp, to_ising
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit.primitives import Sampler, BackendSamplerV2, BackendSampler #samplre is deprecated, but need it to run. Why?

import networkx as nx

import time
import mystic
from mystic.solvers import fmin
from mystic.penalty import quadratic_inequality
from mystic.constraints import as_constraint

import cvxpy as cp

from MaxCutProblem import MaxCutProblem

class Solver():
    """
    Class which contains the ordinary cplex solver.
    Used for getting solutions to compare with quantum solution.
    TODO: Add support for max k-cut
    """
    
    def __init__(self, graph, vertexcover = False, verbose = False, lagrangian = 2):

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


            self.variables = self.model.binary_var_list(len(self.graph), name='x')

            objective = 0

            #for edge in graph.edges:
            self.B = 1
            self.A = lagrangian
            for var in self.variables:
                objective += self.B*var

            #for (i,j) in self.graph.edge_list(): 
             #      self.model.add_constraint(self.A*(1- self.variables[i])*( 1-self.variables[j]) ==0, 'quad') # quadratic constraints to align with ising hamiltonian
            #for (i,j) in self.graph.edge_list(): 
            #    self.model.add_constraint(self.variables[i]+self.variables[j] >= 1) # corresponding non-quadratic constraint
            for (i,j) in self.graph.edge_list():
                objective += self.A*(1- self.variables[i])*( 1-self.variables[j])

            self.objective = objective
            self.model.objective=objective
            self.model.minimize(self.objective)
        else:

            self.graph = graph
            self.model = Model(name="MaxCut")

            self.variables = self.model.binary_var_list(len(self.graph), name='x')
            
            objective = 0

    
            for (i,j, w) in self.graph.weighted_edge_list():            
                objective+= w*(self.variables[i] + self.variables[j] - 2*self.variables[i]*self.variables[j]) 
                # w is a numpy array ( becasue i use np to generate)

            self.objective = objective
            self.model.objective=objective
            self.model.maximize(self.objective)
            
    def evaluate_bitstring(self, bitstring, mark_infeasible = False):
        """
        Evaluates the objective value for a given bitstring.
        """
        if self.vertexcover:
            is_infeasible = 0
            for (i, j) in self.graph.edge_list():
                is_infeasible += self.A*(1 - bitstring[i]) * (1 - bitstring[j])
            if is_infeasible:  #FIXX THIS TODO SOM FAEN
                if mark_infeasible:
                    return (self.B*np.sum(bitstring) + is_infeasible, True) #now returns value + violation
                else:
                    return self.B*np.sum(bitstring) + is_infeasible
            return self.B* np.sum(bitstring)
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
                print(f'Objective to minimize: {self.objective}')
            self.model.minimize(self.objective)
            
            solution = self.model.solve()
            #print('optimal value found:',solution.get_objective_value() )
            bitstring = [var.solution_value for var in self.variables]
            if self.verbose:
                print(solution.get_objective_value(), bitstring)
            return bitstring, solution.get_objective_value()
                        
        if self.verbose:
            print(f'Objective to maximize: {self.objective}')
        self.model.maximize(self.objective)

        solution = self.model.solve()
        bitstring = [var.solution_value for var in self.variables]
        if self.verbose:
            print(solution.get_objective_value(), bitstring)
        return bitstring, solution.get_objective_value()

    def solve_relaxed(self, method):

            if 'method' == 'GW':
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


    def plot_result(self, bitstring):
        """
        Plots graph of partition. Must be run after solve.
        """
        if not bitstring:
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


if __name__ == "__main__":
    # Example usage for a minimum vertex cover problem
    # Create a graph
    # Add edges to form a cycle
    # Generate a random 3-regular graph with 9 nodes
    graph = nx.generators.random_regular_graph(3, 10, seed=42)
    # Convert the NetworkX graph to a PyGraph (rustworkx)
    pygraph = rx.networkx_converter(graph)
    # Create a Gurobi model
    solver = Solver(pygraph, vertexcover=True, verbose=True)
    bitstring, objective_value = solver.solve()
    print("Solution:", bitstring)
    print("Objective value:", objective_value)
