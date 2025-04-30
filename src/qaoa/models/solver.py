from docplex.mp.model import Model
from matplotlib import pyplot as plt
import rustworkx as rx
import numpy as np
from qiskit_optimization.translators import from_docplex_mp


import networkx as nx
import cvxpy as cp

from abc import ABC, abstractmethod

def create_solver(graph, problem_type, **kwargs):
    match problem_type.lower():
        case 'maxcut':
            return MaxCutSolver(graph, problem_type,  **kwargs)
        case 'minvertexcover':
            return MinVertexCoverSolver(graph,problem_type, **kwargs)
        case _:
            raise ValueError(f"Unknown problem type: {problem_type}")

class Solver(ABC):
    """
    Class which contains the ordinary cplex solver.
    Used for getting solutions to compare with quantum solution.
    Needs graph, and what optimization problem it should solve - currently either maxcut or minvertexcover.
    """
    
    def __init__(self, graph, problem_type: str, verbose = False, lagrangian = 2):

        """
        Initializes the model with the given problem, but does not solve.
        Vertexcover is a boolean flag for if the problem is vertexcover, else it is maxcut.
        Lagrangian decides how to weight the constraints, if any."""

        self.graph = graph
        self.problem_type = problem_type
        self.verbose = verbose
        self.lagrangian = lagrangian
                

    def plot_result(self, bitstring=None):
        """
        Plots graph of partition. If no bitstring is supplied, must be run after solve.
        """
        if not bitstring:
            bitstring = [int(var.solution_value) for var in self.variables]

        colors = ["tab:grey" if i == 0 else "tab:purple" for i in bitstring]
        pos, default_axes = rx.spring_layout(self.graph), plt.axes(frameon=True)
        rx.visualization.mpl_draw(self.graph, node_color=colors, node_size=100, alpha=0.8, pos=pos) 


    def __str__(self):
        return f"{self.__class__.__name__} on graph with {len(self.graph)} nodes"

    @abstractmethod
    def evaluate_bitstring(self, bitstring):
        pass

    @abstractmethod
    def build_model(self):
        pass

    def get_qp(self): #TODO: check if this works without explicitly creating the problem unconstrained
        """ Return a quadratic program using : from qiskit_optimization.translators import from_docplex_mp"""
        return from_docplex_mp(self.model) 
    
    @abstractmethod
    def solve(self):
        pass
    
    @abstractmethod
    def solve_relaxed(self, method):
        pass



class MaxCutSolver(Solver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.build_model()
        
    def build_model(self):

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
        objective_value = 0
        for (i, j, w) in self.graph.weighted_edge_list():
            objective_value += w * (bitstring[i] + bitstring[j] - 2 * bitstring[i] * bitstring[j])
        return objective_value
    


    
    def solve(self):
        """
        Solves the problem as it is initialized in the solver.
        Returns bitstring, solution_value
        """
        if self.verbose:
            print(f'Objective to maximize: {self.objective}')


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




class MinVertexCoverSolver(Solver):
        
    def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.build_model()
    
    def build_model(self):

        self.model = Model(name="VertexCover")
        self.variables = self.model.binary_var_list(len(self.graph), name='x')

        objective = 0

        #for edge in graph.edges:
        self.B = 1

        for var in self.variables:
            objective += self.B*var

        for (i,j) in self.graph.edge_list(): #This is quadratic on purpose to make it align with a QUBO. Can be done linearly 
            objective += self.lagrangian*(1- self.variables[i])*( 1-self.variables[j])

        self.objective = objective
        self.model.objective=objective
        self.model.minimize(self.objective)

    def evaluate_bitstring(self, bitstring, mark_infeasible = False):
        """
        Evaluates the objective value for a given bitstring.
        Does so based on what type of problem the solves is initialized for. 
        Mark infeasible is for better plotting of solutions.
        """
        is_infeasible = 0
        for (i, j) in self.graph.edge_list():
            is_infeasible += self.lagrangian*(1 - bitstring[i]) * (1 - bitstring[j])
        if is_infeasible:  #TODO dobbeltsjekk når denne blir kalt
            if mark_infeasible:
                return (self.B*np.sum(bitstring) + is_infeasible, True) #now returns value + violation
            else:
                return self.B*np.sum(bitstring) + is_infeasible
        return self.B* np.sum(bitstring)

    def solve(self):
        """
        Solves the problem as it is initialized in the solver.
        Returns bitstring, solution_value
        """

        if self.verbose:
            print(f'Objective to minimize: {self.objective}')
            
        m = Model(name="vc_exact")
        x = m.binary_var_list(len(self.graph), name="x")
        # cover constraints
        for i,j in self.graph.edge_list():
            m.add_constraint(x[i] + x[j] >= 1)
        # pure size objective
        m.minimize(m.sum(x[i] for i in range(len(self.graph))))
        # Solve the model
        solution = m.solve()
        self.variables = x
        # Extract the solution
        bitstring = [x[i].solution_value for i in range(len(self.graph))]
        
        if self.verbose:
            print(solution.get_objective_value(), bitstring)
            
        return bitstring, solution.get_objective_value()


    def solve_relaxed(self, method = 'GW'):
        """ Solves the relaxed version of a problem, where the X values are continous between 0 and 1. 
        Method keyword is for future use with different relaxed solving methods. Default is Goemanns-Williamson."""

        raise ValueError('Ideal solutions for MVC are half-integral, and therefore not usable for QAOA warm_start.')

