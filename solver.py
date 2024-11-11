import cplex
from docplex.mp.model import Model
from matplotlib import pyplot as plt
from rustworkx import NoEdgeBetweenNodes
import rustworkx as rx
from rustworkx import is_connected
from rustworkx.visualization import mpl_draw as draw_graph
from load_data import load_graph_from_csv

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
            #self.model.parameters.optimalitytarget =3
        else:
            self.variables = self.model.binary_var_list(len(self.graph), name='x')
        
        objective = 0

        for i,var in enumerate(self.variables): #hvorfor går jeg ikke bare gjennom edges??
            for j,var2 in enumerate(self.variables):
                if i != j and i> j:  
                    try:
                        graph.get_edge_data(i,j)
                        objective -= var + var2 - 2*var*var2
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


   
