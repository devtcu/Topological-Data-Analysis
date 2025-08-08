import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from gudhi import AlphaComplex
import networkx as nx
import torch
import torch_geometric
from torch_geometric.utils import from_networkx

def load_data(data_path='nuclei-coordinates.csv', pixel_size=0.1):
    df = pd.read_csv(data_path, header=None)
    coordinates = df.values * pixel_size
    return coordinates

def build_graph(coordinates):
    delaunay = Delaunay(coordinates)
    G = nx.Graph()
    for i, coord in enumerate(coordinates):
        G.add_node(i, pos=(coord[0], coord[1]))
    for simplex in delaunay.simplices:
        for i in range(3):
            G.add_edge(simplex[i], simplex[(i+1)%3])

    local_radius = np.sqrt(10)
    all_dists = np.linalg.norm(coordinates[:, None] - coordinates, axis=2)
    for i in range(len(coordinates)):
        nearby_indices = np.where(all_dists[i] <= local_radius)[0]
        if len(nearby_indices) > 1:
            local_points = coordinates[nearby_indices]
            local_alpha_complex = AlphaComplex(local_points)
            local_simplex_tree = local_alpha_complex.create_simplex_tree()
            local_persistence = local_simplex_tree.persistence()
            local_betti0 = sum(1 for dim, (birth, death) in local_persistence if dim == 0)
            local_betti1 = sum(1 for dim, (birth, death) in local_persistence if dim == 1)
            G.nodes[i]['local_betti0'] = local_betti0
            G.nodes[i]['local_betti1'] = local_betti1
            G.nodes[i]['degree'] = G.degree[i]
        else:
            G.nodes[i]['local_betti0'] = 1
            G.nodes[i]['local_betti1'] = 0
            G.nodes[i]['degree'] = G.degree[i]
    return G, all_dists

def get_local_subgraph(G, node_idx, hops=2):
    ego_graph = nx.ego_graph(G, node_idx, radius=hops)
    nodes = list(ego_graph.nodes())
    return nodes, G.subgraph(nodes)

def prepare_subgraph_data(G, all_dists, subgraph_nodes):
    combined_subgraph = G.subgraph(subgraph_nodes)
    subgraph_data = from_networkx(combined_subgraph)
    subgraph_features = np.array([[combined_subgraph.nodes[i]['local_betti0'], 
                                  combined_subgraph.nodes[i]['local_betti1'], 
                                  combined_subgraph.nodes[i]['degree']] 
                                 for i in combined_subgraph.nodes])
    subgraph_data.x = torch.tensor(subgraph_features, dtype=torch.float)
    
    subgraph_labels = np.zeros(len(subgraph_nodes), dtype=int)
    subgraph_dists = all_dists[subgraph_nodes][:, subgraph_nodes]
    for i, node in enumerate(subgraph_nodes):
        nodes, _ = get_local_subgraph(G, node, hops=2)
        nearby_dists = all_dists[node, nodes]
        syncytial_count = np.sum((nearby_dists < 2) & (nearby_dists > 0))
        total_count = len(nodes) - 1
        if total_count > 0 and syncytial_count / total_count > 0.5:
            subgraph_labels[i] = 1
    
    labeled_mask = np.random.choice([True, False], len(subgraph_nodes), p=[0.1, 0.9])
    subgraph_data.y = torch.tensor(subgraph_labels, dtype=torch.long)
    subgraph_data.train_mask = torch.tensor(labeled_mask, dtype=torch.bool)
    subgraph_data.test_mask = torch.tensor(~labeled_mask, dtype=torch.bool)
    return subgraph_data

def prepare_full_data(G, all_dists):
    full_data = from_networkx(G)
    full_data.x = torch.tensor([[G.nodes[i]['local_betti0'], G.nodes[i]['local_betti1'], G.nodes[i]['degree']] 
                               for i in G.nodes], dtype=torch.float)
    
    full_labels = np.zeros(len(G.nodes), dtype=int)
    for i in range(len(G.nodes)):
        nodes, _ = get_local_subgraph(G, i, hops=2)
        nearby_dists = all_dists[i, nodes]
        syncytial_count = np.sum((nearby_dists < 2) & (nearby_dists > 0))
        total_count = len(nodes) - 1
        if total_count > 0 and syncytial_count / total_count > 0.5:
            full_labels[i] = 1
    full_data.y = torch.tensor(full_labels, dtype=torch.long)
    return full_data