import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.datasets import Planetoid

def draw_degree_distribution(G):
    degree_list = list(dict(G.degree()).values())
    degree_dist = np.bincount(degree_list)
    print("The number of nodes:", G.number_of_nodes())
    print("The number of edges:", G.number_of_edges())
    plt.scatter(list(range(1, len(degree_dist)+1)), degree_dist)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()



def main():
    net_name = 'Pubmed'
    dataset = Planetoid(root='/tmp/' + net_name, name=net_name)
    data = dataset[0]
    edge_index = data.edge_index
    adj_list = edge_index.numpy().T
    G = nx.Graph()
    G.add_edges_from(adj_list)
    draw_degree_distribution(G)

main()