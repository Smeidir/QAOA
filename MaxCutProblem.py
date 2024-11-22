import random
import rustworkx as rx
from rustworkx import is_connected
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
import params


class MaxCutProblem():
    """
    A class for creting the graph which is later converted to a quantum circuit.
    Graph_size: size of graph. Only relevant for random graphs
    create_random: set to true for making random graphs.
    """
    #TODO: make the graphs not have multiple edges between the same nodes
    #TODO: maybe make it faster?

    def __init__(self):
        pass
    
    def get_graph(self,graph_size, create_random = False, random_weights = False, lb= 0, ub = 1):

        if not create_random: 
            graph = rx.PyGraph()
            graph.add_nodes_from(np.arange(0,5,1))
            edge_list = [(0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
            graph.add_edges_from(edge_list)
            return graph
        else: 
            default_weight = 1
            graph = rx.undirected_gnm_random_graph(graph_size, 2*graph_size)
            edge_list = graph.edge_list()
            if random_weights: 
                rng = np.random.default_rng()
                edge_list = [edge+(float(rng.uniform(lb,ub,1)),) for edge in edge_list if (edge[1],edge[0]) not in edge_list] #remove dupes
            else: #kjører ikke fort, er ikke pent, men funker
                edge_list = [edge+(default_weight,) for edge in edge_list if (edge[1],edge[0]) not in edge_list] #remove dupes
            graph.clear_edges()
            graph.add_edges_from(edge_list)
            
            return graph
def read_graph6(self, graph6_str):

    graph = rx.PyGraph.from_graph6(graph6_str)
    draw_graph(graph)

read_graph6
