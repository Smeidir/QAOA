import cplex
from docplex.mp.model import Model
from matplotlib import pyplot as plt
from rustworkx import NoEdgeBetweenNodes
import rustworkx as rx
from rustworkx import is_connected
from rustworkx.visualization import mpl_draw as draw_graph
from load_data import load_graph_from_csv
from cyipopt import minimize_ipopt

class Solver():
    """
    Class which contains the ordinary cplex solver.
    Used for getting solutions to compare with quantum solution.
    TODO: Add support for max k-cut
    """
    
    def __init__(self, graph, relaxed = False):
        self.graph = graph
        self.model = Model(name="MaxCut")
        self.relaxed = relaxed

        if relaxed: 
            self.variables = self.model.continuous_var_list(len(self.graph),lb=0,ub=1, name='x')
            self.model.parameters.optimalitytarget =3
        else:
            self.variables = self.model.binary_var_list(len(self.graph), name='x')
        
        objective = 0

        for i,var in enumerate(self.variables): #hvorfor går jeg ikke bare gjennom edges??
            for j,var2 in enumerate(self.variables):
                if i != j and i> j:  
                    try:
                        graph.get_edge_data(i,j)
                        objective += var + var2 - 2*var*var2
                    except NoEdgeBetweenNodes:
                        pass
        self.objective = objective

    def solve(self):
        
        print(f'Objective to maximize: {self.objective} for relaxed = {self.relaxed}')
        self.model.maximize(self.objective)

        solution = self.model.solve()
        bitstring = [var.solution_value for var in self.variables]
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
solver = Solver(load_graph_from_csv('data/11_nodes_links_scand.csv'), True)
solver.solve()
def max_cut_objective(x, graph):
    objective = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if graph.has_edge(i, j):
                objective += x[i] + x[j] - 2 * x[i] * x[j]
    return -objective  # Minimize the negative to maximize the original objective

def max_cut_constraint(x):
    return x - 1

def solve_with_ipopt(graph):
    x0 = [0.5] * len(graph)  # Initial guess
    bounds = [(0, 1) for _ in range(len(graph))]
    constraints = {'type': 'ineq', 'fun': max_cut_constraint}
    
    result = minimize_ipopt(fun=max_cut_objective, x0=x0, args=(graph,), bounds=bounds, constraints=constraints)
    
    solution_values = result.x
    print(result.fun, solution_values)
    return solution_values, -result.fun

bitstring, objective_value = solve_with_ipopt(load_graph_from_csv('data/11_nodes_links_scand.csv'))
print(f'Objective value: {objective_value}, Bitstring: {bitstring}')

   
