import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import load_data, build_graph, create_coordinate_prediction_data
from higher_order_gcn import HigherOrderGCN
from train import train_model
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(data, model, save_path=None):
    """Create visualization of coordinate prediction results"""
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
    plt.title('Predictions')
    plt.legend()
    plt.axis('equal')
    
    # Plot 3: Error visualization
    plt.subplot(1, 3, 3)
    plt.scatter(true_coords[visible_mask, 0], true_coords[visible_mask, 1], 
               c='blue', alpha=0.6, s=20, label='Visible nodes')
    
    # Show prediction errors as lines
    for i in np.where(hidden_mask)[0]:
        plt.plot([true_coords[i, 0], pred_coords_np[i, 0]], 
                [true_coords[i, 1], pred_coords_np[i, 1]], 
                'r-', alpha=0.7, linewidth=1)
        
    plt.scatter(true_coords[hidden_mask, 0], true_coords[hidden_mask, 1], 
               c='red', alpha=0.8, s=20, label='True positions')
    plt.scatter(pred_coords_np[hidden_mask, 0], pred_coords_np[hidden_mask, 1], 
               c='orange', alpha=0.8, s=20, label='Predicted positions')
    
    plt.title('Prediction Errors')
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Load coordinates and build graph
    print("Loading nuclei coordinates...")
    coordinates = load_data()
    print(f"Loaded {len(coordinates)} nuclei")
    
    print("Building Delaunay graph...")
    G, _ = build_graph(coordinates)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create coordinate prediction data
    data = create_coordinate_prediction_data(coordinates, G)
    
    print("=== Coordinate Prediction Task ===")
    print(f"Total nodes: {data.x.shape[0]}")
    print(f"Visible nodes: {data.visible_mask.sum().item()}")
    print(f"Hidden nodes: {data.hidden_mask.sum().item()}")
    print(f"Edges: {data.edge_index.shape[1]}")
    
    # Initialize model for coordinate prediction
    model = HigherOrderGCN(
        in_channels=data.x.shape[1],
        hidden_channels=64,
        out_channels=2,  # 2D coordinates
        task='coordinate'
    )
    
    # Simple MSE loss for coordinate prediction
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train the model
    print("\nTraining coordinate prediction model...")
    model = train_model(model, data, optimizer, criterion, 
                       num_epochs=100, task='coordinate')
    
    # Final evaluation
    print("\n=== Final Results ===")
    model.eval()
    with torch.no_grad():
        pred_coords = model(data)
        
        # Calculate metrics on hidden nodes
        hidden_pred = pred_coords[data.hidden_mask]
        hidden_true = data.target_coords[data.hidden_mask]
        
        # Distance errors
        distances = torch.norm(hidden_pred - hidden_true, dim=1)
        mean_distance = torch.mean(distances).item()
        median_distance = torch.median(distances).item()
        
        # MSE and MAE
        mse = torch.mean((hidden_pred - hidden_true)**2).item()
        mae = torch.mean(torch.abs(hidden_pred - hidden_true)).item()
        
        print(f"Mean prediction error: {mean_distance:.2f} μm")
        print(f"Median prediction error: {median_distance:.2f} μm")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        
        # Compare to random baseline
        coords_range = data.target_coords.max(0)[0] - data.target_coords.min(0)[0]
        random_baseline = torch.mean(coords_range).item() / 3.0  # Rough estimate
        print(f"Improvement over random (~{random_baseline:.1f} μm): {((random_baseline - mean_distance) / random_baseline * 100):.1f}%")
    
    # Create visualization
    print("\nGenerating visualization...")
    visualize_results(data, model, 'coordinate_prediction_results.png')
    
    return model, data

if __name__ == "__main__":
    model, data = main() 