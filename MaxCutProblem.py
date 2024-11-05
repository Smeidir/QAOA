import random
import rustworkx as rx
from rustworkx import is_connected
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np


class MaxCutProblem():
    """
    A class for creting the graph which is later converted to a quantum circuit.
    Graph_size: size of graph. Only relevant for random graphs
    create_random: set to true for making random graphs.
    """

    def __init__(self, graph_size, create_random = False):
        pass
    
    def get_graph(self,graph_size, create_random = False):

        if not create_random: 
            graph = rx.PyGraph()
            graph.add_nodes_from(np.arange(0,5,1))
            edge_list = [(0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
            graph.add_edges_from(edge_list)
            return graph
        else: 
            default_weight = 1
            graph = rx.PyGraph()
            rng = np.random.default_rng(seed = 173)
            graph.add_nodes_from(np.arange(0,graph_size,1))
            edge_list= [(random.randint(0,graph_size-1),random.randint(0,graph_size-1),1) for _ in range(2*graph_size)]
            
            edge_list = [edge for edge in edge_list if (edge[1],edge[0],1.0) not in edge_list] #remove dupes
            
            graph.add_edges_from(edge_list)

            while not is_connected(graph):
                #naive function iteratively adds edges until the graph is connected. can be immensely improved but runs fast enough
                edge = (random.randint(0,graph_size-1),random.randint(0,graph_size-1),1)
                mirror_edge = (edge[1],edge[0],1.0)

                if edge not in graph.edge_list() and mirror_edge not in graph.edge_list(): #adding, but avoiding dupes
                    graph.add_edges_from([(random.randint(0,graph_size-1),random.randint(0,graph_size-1),1) for _ in range(1)])

            return graph
