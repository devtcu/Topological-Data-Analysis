import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from data_utils import load_data, build_graph, create_coordinate_prediction_data
from contrastive_gcn import ContrastiveGCN
from contrastive_utils import train_contrastive_model, evaluate_contrastive_model
from higher_order_gcn import HigherOrderGCN
from train import train_model

def visualize_contrastive_results(data, model, save_path=None):
    """Create visualization of contrastive coordinate prediction results"""
    model.eval()
    with torch.no_grad():
        pred_coords = model(data)
    
    # Get coordinates and predictions
    true_coords = data.target_coords.numpy()
    pred_coords_np = pred_coords.numpy()
    
    visible_mask = data.visible_mask.numpy()
    hidden_mask = data.hidden_mask.numpy()
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original data with hidden nodes shown
    plt.subplot(1, 3, 1)
    plt.scatter(true_coords[visible_mask, 0], true_coords[visible_mask, 1], 
               c='blue', alpha=0.6, s=20, label='Visible nodes')
    plt.scatter(true_coords[hidden_mask, 0], true_coords[hidden_mask, 1], 
               c='red', alpha=0.8, s=20, label='Hidden nodes (ground truth)')
    plt.title('Ground Truth')
    plt.legend()
    plt.axis('equal')
    
    # Plot 2: Predicted coordinates
    plt.subplot(1, 3, 2)
    plt.scatter(true_coords[visible_mask, 0], true_coords[visible_mask, 1], 
               c='blue', alpha=0.6, s=20, label='Visible nodes')
    plt.scatter(pred_coords_np[hidden_mask, 0], pred_coords_np[hidden_mask, 1], 
               c='orange', alpha=0.8, s=20, label='Predicted positions')
    plt.title('Contrastive GCN Predictions')
    plt.legend()
    plt.axis('equal')
    
    # Plot 3: Error visualization
    plt.subplot(1, 3, 3)
    plt.scatter(true_coords[visible_mask, 0], true_coords[visible_mask, 1], 
               c='blue', alpha=0.3, s=10, label='Visible nodes')
    
    # Show prediction errors as lines
    for i in np.where(hidden_mask)[0]:
        plt.plot([true_coords[i, 0], pred_coords_np[i, 0]], 
                [true_coords[i, 1], pred_coords_np[i, 1]], 
                'r-', alpha=0.5, linewidth=0.5)
        
    plt.scatter(true_coords[hidden_mask, 0], true_coords[hidden_mask, 1], 
               c='red', alpha=0.8, s=20, label='True positions')
    plt.scatter(pred_coords_np[hidden_mask, 0], pred_coords_np[hidden_mask, 1], 
               c='orange', alpha=0.8, s=20, label='Predicted positions')
    
    plt.title('Prediction Errors')
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main_contrastive():
    print("=== CONTRASTIVE LEARNING FOR COORDINATE PREDICTION ===")
    print("Task: Predict coordinates of hidden nuclei using contrastive learning")
    print()
    
    # Load coordinates and build graph
    print("Loading nuclei coordinates...")
    coordinates = load_data()
    print(f"Loaded {len(coordinates)} nuclei")
    
    print("Building Delaunay graph...")
    G, _ = build_graph(coordinates)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create coordinate prediction data
    data = create_coordinate_prediction_data(coordinates, G)
    
    print("=== Contrastive Coordinate Prediction Task ===")
    print(f"Total nodes: {data.x.shape[0]}")
    print(f"Visible nodes: {data.visible_mask.sum().item()}")
    print(f"Hidden nodes: {data.hidden_mask.sum().item()}")
    print(f"Edges: {data.edge_index.shape[1]}")
    
    # Initialize contrastive model
    contrastive_model = ContrastiveGCN(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        proj_dim=64,
        tau=0.5
    )
    
    print(f"Contrastive model parameters: {sum(p.numel() for p in contrastive_model.parameters()):,}")
    
    # Initialize standard model for comparison
    standard_model = HigherOrderGCN(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        out_channels=2,
        task='coordinate'
    )
    
    print(f"Standard model parameters: {sum(p.numel() for p in standard_model.parameters()):,}")
    
    # Setup training
    contrastive_optimizer = optim.Adam(contrastive_model.parameters(), lr=0.001, weight_decay=1e-5)
    standard_optimizer = optim.Adam(standard_model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Train contrastive model
    print("\n=== Training Contrastive Model ===")
    contrastive_model, _ = train_contrastive_model(
        contrastive_model, data, contrastive_optimizer, criterion, 
        num_epochs=100, lambda_contrast=0.1
    )
    
    # Train standard model
    print("\n=== Training Standard Model ===")
    standard_model = train_model(
        standard_model, data, standard_optimizer, criterion, 
        num_epochs=100, task='coordinate'
    )
    
    # Evaluate both models
    print("\n=== Evaluation ===")
    contrastive_results = evaluate_contrastive_model(contrastive_model, data)
    
    # Evaluate standard model
    standard_model.eval()
    with torch.no_grad():
        pred_coords_std = standard_model(data)
        
        # Calculate metrics on hidden nodes
        hidden_pred_std = pred_coords_std[data.hidden_mask]
        hidden_true_std = data.target_coords[data.hidden_mask]
        
        # Distance errors
        distances_std = torch.norm(hidden_pred_std - hidden_true_std, dim=1)
        mean_distance_std = torch.mean(distances_std).item()
        median_distance_std = torch.median(distances_std).item()
        
        # MSE and MAE
        mse_std = torch.mean((hidden_pred_std - hidden_true_std)**2).item()
        mae_std = torch.mean(torch.abs(hidden_pred_std - hidden_true_std)).item()
    
    # Print results comparison
    print("\n=== Results Comparison ===")
    print(f"Standard GCN Mean Distance Error: {mean_distance_std:.2f} μm")
    print(f"Contrastive GCN Mean Distance Error: {contrastive_results['mean_distance_error']:.2f} μm")
    
    improvement = (mean_distance_std - contrastive_results['mean_distance_error']) / mean_distance_std * 100
    print(f"Improvement with Contrastive Learning: {improvement:.1f}%")
    
    # Compare to random baseline
    coords_range = data.target_coords.max(0)[0] - data.target_coords.min(0)[0]
    random_baseline = torch.mean(coords_range).item() / 3.0  # Rough estimate
    
    print(f"\nComparison to Random Baseline (~{random_baseline:.1f} μm):")
    print(f"Standard GCN: {((random_baseline - mean_distance_std) / random_baseline * 100):.1f}% better than random")
    print(f"Contrastive GCN: {((random_baseline - contrastive_results['mean_distance_error']) / random_baseline * 100):.1f}% better than random")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_contrastive_results(data, contrastive_model, 'contrastive_predictions.png')
    
    return contrastive_model, standard_model, data

if __name__ == "__main__":
    contrastive_model, standard_model, data = main_contrastive()
