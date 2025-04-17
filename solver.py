from docplex.mp.model import Model
from matplotlib import pyplot as plt
import rustworkx as rx
import numpy as np
from qiskit_optimization.translators import from_docplex_mp, to_ising
from qiskit_algorithms.optimizers import COBYLA

import networkx as nx
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
        Lagrangian decides how to weight the constraints, if any.
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

            for (i,j) in self.graph.edge_list(): #This is quadratic on purpose to make it align with a QUBO. Can be done linear 
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

            self.objective = objective
            self.model.objective=objective
            self.model.maximize(self.objective)
            
    def evaluate_bitstring(self, bitstring, mark_infeasible = False):
        """
        Evaluates the objective value for a given bitstring.
        Does so based on what type of problem the solves is initialized for. 
        Mark infeasible is for better plotting of solutions.
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
        Solves the problem as it is initialized in the solver.
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

    def solve_relaxed(self, method = 'GW'):
            """ Solves the relaxed version of a problem, where the X values are continous between 0 and 1. 
            Method keyword is for future use with different relaxed solving methods. Default is Goemanns-Williamson."""

            if method == 'GW':
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
        Plots graph of partition. If no bitstring is supplied, must be run after solve.
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
