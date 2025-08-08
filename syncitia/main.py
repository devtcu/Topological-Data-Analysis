import numpy as np
from higher_order_gcn import HigherOrderGCN
from data_utils import load_data, build_graph, get_local_subgraph, prepare_subgraph_data, prepare_full_data
from vis_utils import compute_alpha_number, plot_voronoi, plot_predictions
from train import train_model, evaluate_model, get_feature_importance
import torch
import pickle

def main():
    print("Loading data...")
    coordinates = load_data()
    n_nodes = len(coordinates)

    print("Building graph...")
    G, all_dists = build_graph(coordinates)
    alpha_number = compute_alpha_number(coordinates)
    print(f"Alpha Number: {alpha_number:.2f} µm²")
    plot_voronoi(coordinates)

    print("Selecting subgraphs...")
    degrees = np.array([G.degree[i] for i in range(n_nodes)])
    subgraph_centers = [np.argmax(degrees), np.argmin(degrees), np.random.randint(0, n_nodes)]
    all_subgraph_nodes = []
    for center in subgraph_centers:
        nodes, _ = get_local_subgraph(G, center, hops=2)
        all_subgraph_nodes.extend(nodes)
    all_subgraph_nodes = list(set(all_subgraph_nodes))
    print(f"Combined subgraph has {len(all_subgraph_nodes)} nodes")

    print("Preparing data...")
    subgraph_data = prepare_subgraph_data(G, all_dists, all_subgraph_nodes)
    full_data = prepare_full_data(G, all_dists)

    print("Training GCN...")
    model = HigherOrderGCN(in_channels=3, hidden_channels=16, out_channels=2, max_order=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model = train_model(model, subgraph_data, optimizer, criterion)

    print("Predicting surroundings...")
    predictions, full_acc = evaluate_model(model, full_data)
    print(f"Full Graph Test Accuracy: {full_acc:.4f}")
    plot_predictions(coordinates, predictions)

    feature_importance = get_feature_importance(model)
    print(f"Feature Importance: {', '.join(f'{name}: {value:.3f}' for name, value in feature_importance.items())}")

    with open('nuclei_graph.pkl', 'wb') as f:
        pickle.dump(G, f)

    print("\nSummary:")
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Average node degree: {np.mean(list(dict(G.degree()).values())):.2f}")
    print(f"Alpha Number: {alpha_number:.2f} µm²")
    print(f"Proportion of syncytial-dominated surroundings: {np.mean(predictions):.2%}")
    print(f"Full Graph Test Accuracy: {full_acc:.4f}")

if __name__ == "__main__":
    main()