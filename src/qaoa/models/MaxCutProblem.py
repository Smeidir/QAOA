import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
import os
import networkx as nx
from matplotlib import pyplot as plt


class MaxCutProblem():
    """
    A class for creting the graph which is later converted to a quantum circuit.
    Graph_size: size of graph. Only relevant for random graphs
    create_random: set to true for making random graphs.
    """
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
        
    def get_complete_graphs(self,sizes):
        return [rx.generators.complete_graph(size) for size in sizes]
    
    def get_test_graphs(self, n = 5):
        graph_dir = 'graphs'
        graphs = []
        names = []
        rng = np.random.default_rng(seed= 1373)
        for filename in os.listdir(graph_dir):
            identifier = 'saved_graph' + str(n)
            if filename.startswith(identifier):
                with open(os.path.join(graph_dir, filename), 'r') as file:
                    graph6_str = file.read().strip()
                    graphs_array = graph6_str.split()
                    for graph_str in graphs_array:
                        graph = nx.from_graph6_bytes(graph_str.encode())
                        pygraph = rx.PyGraph()
                        node_mapping = {node: pygraph.add_node(node) for node in graph.nodes}

                        # Add edges, including weights if present
                        for u, v, data in graph.edges(data=True):
                            weight = data.get("weight", rng.uniform(0,1))  
                            pygraph.add_edge(node_mapping[u], node_mapping[v], weight)

                        names.append(graph_str)
                        graphs.append(pygraph)
        return graphs, names
    
    def get_graph_by_name(self, name):
        graphs, names = self.get_test_graphs()
        all_graphs, all_names = [], []
        for n in range(5, 10):  # Assuming n ranges from 5 to 9
            graphs, names = self.get_test_graphs(n)
            all_graphs.extend(graphs)
            all_names.extend(names)

        graph = all_graphs[all_names.index(name)]
        return graph


    def get_single_graphs(self):
        graphs,names = self.get_test_graphs(5)
        graph5 = 'D~{'
        graph9 = 'HCrfZzf'
        graph5_graph = graphs[names.index(graph5)]
        graphs,names = self.get_test_graphs(9)
        graph9_graph = graphs[names.index(graph9)]

  
        return [graph9_graph, graph5_graph], [graph5, graph9]
        
    def draw_test_graphs(self, n=5):
        graphs, names = self.get_test_graphs(n)
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, graph in enumerate(graphs):
            ax = axes[i]
            draw_graph(graph, ax=ax)
            ax.set_title(f"Graph {i+1} with graph6: {names[i]}")
        
        plt.tight_layout()
        plt.show()


    def get_erdos_renyi_graphs(self,sizes):
        #TODO: only replicatable if the same sizes are given in the same order.
        seed = 40
        #for 5,7,9 the seed waas 40 and probabilities 0.35 and 0.7

        rng = np.random.default_rng(seed)
        graphs = []

        for size in sizes:
            graph_sparse = rx.undirected_gnp_random_graph(size, 0.35, seed=seed)
            edge_list_sparse = graph_sparse.edge_list()
            edge_list_sparse = [edge + (float(rng.choice([0.25, 0.5, 0.75, 1])),) for edge in edge_list_sparse]
            graph_sparse.clear_edges()
            graph_sparse.add_edges_from(edge_list_sparse)
            graphs.append(graph_sparse)

            graph_dense = rx.undirected_gnp_random_graph(size, 0.7, seed=seed)

            edge_list_dense = graph_dense.edge_list()
            edge_list_dense = [edge + (float(rng.choice([0.25, 0.5, 0.75, 1])),) for edge in edge_list_dense]
            graph_dense.clear_edges()
            graph_dense.add_edges_from(edge_list_dense)
            graphs.append(graph_dense)
        return graphs
    

    def get_erdos_renyi_graphs_paper1(self):
        """This is a pretty specific function for giving 4 graphs, 3 dense and one sparse, of sizes 6, 9 and 12, where the sparse
        one is size 9. That's why it's pretty hardcoded. Gives in order dense 6,9,12, sparse 9.
        """
        #TODO: only replicatable if the same sizes are given in the same order.
        seed = 40
        #This is suppos

        rng = np.random.default_rng(seed)
        graphs = []

        for size in [6,9,12]:
            

            graph_dense = rx.undirected_gnp_random_graph(size, 0.6, seed=seed)

            #edge_list_dense = graph_dense.edge_list()
            #edge_list_dense = [edge + (float(rng.choice([0.25, 0.5, 0.75, 1])),) for edge in edge_list_dense]
            #graph_dense.clear_edges()
            #graph_dense.add_edges_from(edge_list_dense)
            graphs.append(graph_dense)

        for size in [9]:
            graph_sparse = rx.undirected_gnp_random_graph(size, 0.3, seed=seed)
            #edge_list_sparse = graph_sparse.edge_list()
            #edge_list_sparse = [edge + (float(rng.choice([0.25, 0.5, 0.75, 1])),) for edge in edge_list_sparse]
            #graph_sparse.clear_edges()
            #graph_sparse.add_edges_from(edge_list_sparse)
            graphs.append(graph_sparse)

        for graph in graphs:
            for n in graph.node_indices():
                graph[n] = rng.uniform(0.25, 1)

        #graphs = [graphs[1], graphs[3]]
        return graphs
    def draw_given_graphs(self, graph_names):

        graphs, names = self.get_test_graphs()
        all_graphs, all_names = [], []
        for n in range(5, 10):  # Assuming n ranges from 5 to 9
            graphs, names = self.get_test_graphs(n)
            all_graphs.extend(graphs)
            all_names.extend(names)
        
        num_graphs = len(graph_names)
        num_cols = 4
        num_rows = (num_graphs + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
        axes = axes.flatten()
        
        for i, graph_name in enumerate(graph_names):
            graph = all_graphs[all_names.index(graph_name)]
            ax = axes[i]
            draw_graph(graph, ax=ax)
            ax.set_title(f"Graph with graph6: {graph_name}")
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()


    def get_representative_graphs(self):
        complete_graph = rx.generators.complete_graph(7)
        sparse_graph = rx.undirected_gnm_random_graph(7, 9)
        dense_graph = rx.undirected_gnm_random_graph(7,18)



        rng = np.random.default_rng(seed=13)
        edge_list_complete = complete_graph.edge_list()
        edge_list_sparse = sparse_graph.edge_list()
        edge_list_dense = dense_graph.edge_list()


        edge_list_complete = [edge+(float(rng.uniform(0,1,1)),) for edge in edge_list_complete]
        edge_list_sparse = [edge+(float(rng.uniform(0,1,1)),) for edge in edge_list_sparse if (edge[1],edge[0]) not in edge_list_sparse]
        edge_list_dense = [edge+(float(rng.uniform(0,1,1)),) for edge in edge_list_dense if (edge[1],edge[0]) not in edge_list_dense]

        complete_graph.clear_edges()
        complete_graph.add_edges_from(edge_list_complete)

        sparse_graph.clear_edges()
        sparse_graph.add_edges_from(edge_list_sparse)

        dense_graph.clear_edges()
        dense_graph.add_edges_from(edge_list_dense)


        self.complete_graph = complete_graph
        self.sparse_graph = sparse_graph
        self.dense_graph = dense_graph
        return complete_graph, sparse_graph, dense_graph
        
        
    def draw_representative_graphs(self):
        graphs = self.get_representative_graphs()
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        axes = axes.flatten()
        
        for i, graph in enumerate(graphs): 
            print(i)
            ax = axes[i]
            draw_graph(graph, ax=ax)
            ax.set_title(f"Representative Graph {i+1}")
        
        plt.tight_layout()
        plt.show()

    def print_graphs(self):

        graphs = [self.complete_graph, self.sparse_graph, self.dense_graph]
        graph_names = ["Complete Graph", "Sparse Graph", "Dense Graph"]
        
        for graph, name in zip(graphs, graph_names):
            nx_graph = nx.Graph()
            nx_graph.add_nodes_from(range(graph.num_nodes()))
            nx_graph.add_weighted_edges_from(graph.weighted_edge_list())
            graph6_str = nx.to_graph6_bytes(nx_graph).decode().strip()
            print(f"{name} (graph6): {graph6_str}")
    def load_representative_graphs(self):

        rng = np.random.default_rng(seed=13)
        edge_list_complete = complete_graph.edge_list()
        edge_list_sparse = sparse_graph.edge_list()
        edge_list_dense = dense_graph.edge_list()


        edge_list_complete = [edge+(float(rng.uniform(0,1,1)),) for edge in edge_list_complete]
        edge_list_sparse = [edge+(float(rng.uniform(0,1,1)),) for edge in edge_list_sparse if (edge[1],edge[0]) not in edge_list_sparse]
        edge_list_dense = [edge+(float(rng.uniform(0,1,1)),) for edge in edge_list_dense if (edge[1],edge[0]) not in edge_list_dense]

        complete_graph.clear_edges()
        complete_graph.add_edges_from(edge_list_complete)

        sparse_graph.clear_edges()
        sparse_graph.add_edges_from(edge_list_sparse)

        dense_graph.clear_edges()
        dense_graph.add_edges_from(edge_list_dense)

        return complete_graph, sparse_graph, dense_graph
    


    def get_paper_graphs(self):
        edge_list_1 = [(1,5),(2,4),(3,4),(3,5),(4,5)]
        edge_list_2 = [(1,4),(1,5), (2,3),(2,5),(3,5),(4,5)]
        edge_list_3 = [(1,3),(1,4),(1,5), (2,3),(2,4),(2,5), (3,5), (4,5)]
        graph1 = rx.PyGraph()
        graph1.add_nodes_from(range(1, 6))
        graph1.add_edges_from([(u-1, v-1, 1.0) for u, v in edge_list_1])

        graph2 = rx.PyGraph()
        graph2.add_nodes_from(range(1, 6))
        graph2.add_edges_from([(u-1, v-1, 1.0) for u, v in edge_list_2])

        graph3 = rx.PyGraph()
        graph3.add_nodes_from(range(1, 6))
        graph3.add_edges_from([(u-1, v-1, 1.0) for u, v in edge_list_3])

        return [graph1, graph2, graph3]
    
    

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

            if filename == 'graph5c.g6.txt': #not enough symmetric:
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


                    elif not (is_odd or is_even) and asymmetric <16:
                        graphs_to_save.append(graphs_array[index])
                        asymmetric +=1


                    #print(f"Degree parities: {degree_parities}, Is odd: {is_odd}, Is even: {is_even}")

                print(f"File: {filename}, Asymmetric: {asymmetric},Symmetric: {symmetric}")
                with open(os.path.join(graph_dir, f'saved_{filename}'), 'w') as save_file:
                    save_file.write('\n'.join(graphs_to_save))

            
            else: 
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


