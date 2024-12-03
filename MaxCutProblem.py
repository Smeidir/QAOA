import random
import rustworkx as rx
from rustworkx import is_connected
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
import params
import os
import networkx as nx
from matplotlib import pyplot as plt


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
    def get_test_graphs(self, n = 5):
        graph_dir = 'graphs'
        graphs = []
        rng = np.random.default_rng()
        for filename in os.listdir(graph_dir):
            identifier = 'saved_graph' + str(n)
            if filename.startswith(identifier):
                with open(os.path.join(graph_dir, filename), 'r') as file:
                    graph6_str = file.read().strip()
                    graphs_array = graph6_str.split()
                    for graph_str in graphs_array:
                        graph = nx.from_graph6_bytes(graph_str.encode())
                        edge_list = [edge+(float(rng.uniform(0,1,1)),) for edge in graph.edges]
                        graph.edges = edge_list
                        graphs.append(graph)
            return graphs

def save_graphs(): #Code for getting the graphs from public directory: https://users.cecs.anu.edu.au/~bdm/data/graphs.html
    graph_dir = 'graphs'
    for filename in os.listdir(graph_dir):

        with open(os.path.join(graph_dir, filename), 'r') as file:
            graph6_str = file.read().strip()
            graphs_array = graph6_str.split()
            
            symmetric = 0
            asymmetric = 0
            graphs_to_save = []

            indices = np.arange(len(graphs_array))
            np.random.shuffle(indices)
            

            for index in indices:
                graph = nx.from_graph6_bytes(graphs_array[index].encode())

                degree_parities =[graph.degree(n)%2 for n in graph.nodes]  

                is_odd= np.all(degree_parities)
                is_even = not np.any(degree_parities)

                if symmetric +  asymmetric == 20:
                    break

                if (is_odd or is_even) and symmetric <10:
                    graphs_to_save.append(graphs_array[index])
                    symmetric += 1


                elif not (is_odd or is_even) and asymmetric <10:
                    graphs_to_save.append(graphs_array[index])
                    asymmetric +=1


                #print(f"Degree parities: {degree_parities}, Is odd: {is_odd}, Is even: {is_even}")

            print(f"File: {filename}, Asymmetric: {asymmetric},Symmetric: {symmetric}")
            with open(os.path.join(graph_dir, f'saved_{filename}'), 'w') as save_file:
                save_file.write('\n'.join(graphs_to_save))


