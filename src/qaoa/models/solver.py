from docplex.mp.model import Model
from matplotlib import pyplot as plt
import rustworkx as rx
import numpy as np
from qiskit_optimization.translators import from_docplex_mp

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

        try:

            for (i,j, w) in self.graph.weighted_edge_list():            
                objective+= w*(self.variables[i] + self.variables[j] - 2*self.variables[i]*self.variables[j]) 
        except AttributeError:
            for (i,j) in self.graph.edge_list():            
                objective+= (self.variables[i] + self.variables[j] - 2*self.variables[i]*self.variables[j]) 

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





class MinVertexCoverSolver(Solver):

    """WARNING: MUST BE AWEIGHTED GRAPH!
    If not, we will have bugs with the weights - an unweighted graph will be weighted by index.
    THis is obviously a todo but i dont have all the time in the world."""
        
    def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.build_model()
    
    def build_model(self):

        self.model = Model(name="VertexCover")
        self.variables = self.model.binary_var_list(len(self.graph), name='x')

        objective = 0

        #for edge in graph.edges:
        self.B = 1

        for i,var in enumerate(self.variables):
            objective += self.B*var*self.graph[i]

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
        obj_value = self.B*sum([bitstring[i]*self.graph[i] for i in range(len(bitstring))])

        for (i, j) in self.graph.edge_list():
            is_infeasible += self.lagrangian*(1 - bitstring[i]) * (1 - bitstring[j])
        if is_infeasible:  #TODO dobbeltsjekk når denne blir kalt
            if mark_infeasible:
                return (obj_value + is_infeasible, True) #now returns value + violation
            else:
                return obj_value + is_infeasible
        if mark_infeasible:
            return (obj_value, False)
        return obj_value

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
        m.minimize(m.sum(x[i]*self.graph[i] for i in range(len(self.graph))))
        # Solve the model
        solution = m.solve()
        self.variables = x
        # Extract the solution
        bitstring = [x[i].solution_value for i in range(len(self.graph))]
        
        if self.verbose:
            print(solution.get_objective_value(), bitstring)
            
        return bitstring, solution.get_objective_value()

    def get_feasible_solutions_hamming(self):
        """
        Returns all feasible solutions to the Minimum Vertex Cover problem.
        A solution is feasible if every edge has at least one of its endpoints in the cover.
        Not optimized at all.

        Returns:
            dict: A list of all feasible bitstrings (vertex covers), sorted by hamming distance from the best one.
        """
        n = len(self.graph)
        feasible_solutions = {}
        # Generate all possible bitstrings
        for i in range(2**n):
            # Convert integer to binary representation (bitstring)
            bitstring = [(i >> j) & 1 for j in range(n)]
            
            # Check if this is a valid vertex cover
            is_valid = True
            for u, v in self.graph.edge_list():
                if bitstring[u] == 0 and bitstring[v] == 0:
                    is_valid = False
                    break
            if is_valid:
                try:
                    feasible_solutions[self.evaluate_bitstring(bitstring)].append(bitstring)
                    
                except KeyError:
                    feasible_solutions[self.evaluate_bitstring(bitstring)] = [bitstring]
        best_solution = feasible_solutions[min(feasible_solutions.keys())]
        hamming_dict = {x: [] for x in range(n)}
        for current_solution in feasible_solutions.values():
            hamming_dict[np.count_nonzero(np.array(best_solution)!=np.array(current_solution))].append(current_solution[0])



                    
        return hamming_dict

    def solve_relaxed(self, method = 'GW'):
        """ Solves the relaxed version of a problem, where the X values are continous between 0 and 1. 
        Method keyword is for future use with different relaxed solving methods. Default is Goemanns-Williamson."""

        raise ValueError('Ideal solutions for MVC are half-integral, and therefore not usable for QAOA warm_start.')
"""
def main():
    
    #Example usage of the MinVertexCoverSolver to find and visualize all feasible solutions.
    
    import matplotlib.pyplot as plt
    
    # Create a simple graph
    # Get a graph from paper1
    problem = MaxCutProblem()
    # Get one of the graphs from the paper collection
    paper_graphs = problem.get_erdos_renyi_graphs_paper1()
    G = paper_graphs[0]  # Using the first graph from the collection
    
    # Create the solver
    solver = MinVertexCoverSolver(G, "MinVertexCover")
    test_solve = solver.solve()
    
    # Get all feasible solutions
    feasible_solutions = solver.get_feasible_solutions()
    # Print all solutions by size
    total_solutions = sum(len(solutions) for solutions in feasible_solutions.values())
    print(f"Found {total_solutions} feasible vertex covers of graph of size {len(G.nodes())}, sparse:")
    
    all_solutions = []
    for size, solutions in sorted(feasible_solutions.items()):
        print(f"Size {size}: {len(solutions)} solutions")
        for solution in solutions:
            all_solutions.append(solution)
    
    if not all_solutions:
        print("No feasible solutions found.")
        return
    


    # Visualize the graph and solutions
    import matplotlib.pyplot as plt

    # Show solutions in a scrollable manner
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)
    
    pos = rx.spring_layout(G)
    current_idx = [0]  # Using list to make it mutable in nested functions
    
    def draw_solution(idx):
        ax.clear()
        solution = all_solutions[idx]
        colors = ["tab:red" if node == 1 else "tab:grey" for node in solution]
        
        rx.visualization.mpl_draw(G, node_color=colors, node_size=400, 
                                pos=pos, ax=ax, with_labels=True)
        ax.set_title(f"Solution {idx+1}/{len(all_solutions)}: Size = {sum(solution)}")
    
    def next_solution(event):
        current_idx[0] = (current_idx[0] + 1) % len(all_solutions)
        draw_solution(current_idx[0])
        plt.draw()
    
    def prev_solution(event):
        current_idx[0] = (current_idx[0] - 1) % len(all_solutions)
        draw_solution(current_idx[0])
        plt.draw()
    
    # Create buttons for navigation
    ax_prev = plt.axes([0.1, 0.05, 0.15, 0.075])
    ax_next = plt.axes([0.55, 0.05, 0.15, 0.075])
    
    b_next = Button(ax_next, 'Next')
    b_next.on_clicked(next_solution)
    b_prev = Button(ax_prev, 'Previous')
    b_prev.on_clicked(prev_solution)
    
    # Draw initial solution
    draw_solution(current_idx[0])

    
    
    plt.tight_layout()
    plt.show()


    def solve_relaxed(self, method = 'GW'):
       Solves the relaxed version of a problem, where the X values are continous between 0 and 1. 
        Method keyword is for future use with different relaxed solving methods. Default is Goemanns-Williamson.

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
if __name__ == "__main__":
    main()"""