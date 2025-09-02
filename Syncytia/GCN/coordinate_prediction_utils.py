import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def create_missing_node_task(coordinates, G, missing_fraction=0.3, min_cluster_size=5):
    """
    Create a meaningful coordinate prediction task by hiding nodes strategically
    """
    n_nodes = len(coordinates)
    
    # Strategy: Hide connected clusters rather than random nodes
    # This makes the task harder and more realistic
    
    # Find connected components and select some to hide
    import networkx as nx
    
    # Create spatial clusters to hide
    from sklearn.cluster import KMeans
    n_clusters = max(3, int(n_nodes * missing_fraction / 50))  # ~50 nodes per cluster
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(coordinates)
    
    # Select some clusters to hide completely
    unique_clusters = np.unique(cluster_labels)
    n_hidden_clusters = max(1, len(unique_clusters) // 3)
    hidden_clusters = np.random.choice(unique_clusters, n_hidden_clusters, replace=False)
    
    # Nodes to hide
    hidden_mask = np.isin(cluster_labels, hidden_clusters)
    hidden_indices = np.where(hidden_mask)[0]
    visible_indices = np.where(~hidden_mask)[0]
    
    print(f"Hiding {len(hidden_indices)} nodes ({len(hidden_indices)/n_nodes*100:.1f}%) in {n_hidden_clusters} clusters")
    print(f"Keeping {len(visible_indices)} nodes visible for training")
    
    return visible_indices, hidden_indices, hidden_mask

def prepare_coordinate_data(coordinates, G, visible_indices, hidden_indices):
    """
    Prepare data for coordinate prediction task
    """
    n_nodes = len(coordinates)
    
    # Create masks
    visible_mask = torch.zeros(n_nodes, dtype=torch.bool)
    hidden_mask = torch.zeros(n_nodes, dtype=torch.bool)
    visible_mask[visible_indices] = True
    hidden_mask[hidden_indices] = True
    
    # Node features: [local_betti0, local_betti1, degree, is_visible, x_coord, y_coord]
    # For hidden nodes, we'll mask the coordinates with zeros
    node_features = []
    
    for i in range(n_nodes):
        # Topological features (always available)
        neighbors = list(G.neighbors(i))
        degree = len(neighbors)
        
        # Simple local topology (you can make this more sophisticated)
        local_betti0 = min(degree, 10) / 10.0  # Normalized
        local_betti1 = max(0, degree - 6) / 10.0  # Rough estimate
        
        # Visibility flag
        is_visible = 1.0 if visible_mask[i] else 0.0
        
        # Coordinates (masked for hidden nodes)
        if visible_mask[i]:
            x_coord = coordinates[i, 0]
            y_coord = coordinates[i, 1]
        else:
            x_coord = 0.0  # Hidden
            y_coord = 0.0  # Hidden
        
        node_features.append([local_betti0, local_betti1, degree/20.0, is_visible, x_coord/100.0, y_coord/100.0])
    
    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float32)
    target_coords = torch.tensor(coordinates, dtype=torch.float32)
    
    # Edge index (keep all edges - this is key for information propagation)
    edge_index = torch.tensor([[u, v] for u, v in G.edges()], dtype=torch.long).t().contiguous()
    
    return {
        'x': x,
        'edge_index': edge_index,
        'target_coords': target_coords,
        'visible_mask': visible_mask,
        'hidden_mask': hidden_mask,
        'visible_indices': visible_indices,
        'hidden_indices': hidden_indices
    }

def train_coordinate_model(model, data, optimizer, num_epochs=200):
    """
    Train the coordinate prediction model
    """
    model.train()
    
    # Loss functions
    coord_criterion = torch.nn.MSELoss()
    confidence_criterion = torch.nn.BCELoss()
    
    train_losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        pred_coords, pred_confidence = model(data['x'], data['edge_index'])
        
        # Coordinate loss (only on hidden nodes - this is the key!)
        coord_loss = coord_criterion(pred_coords[data['hidden_mask']], 
                                   data['target_coords'][data['hidden_mask']])
        
        # Confidence loss (hidden nodes should have low confidence, visible high)
        target_confidence = data['visible_mask'].float()
        confidence_loss = confidence_criterion(pred_confidence, target_confidence)
        
        # Combined loss
        total_loss = coord_loss + 0.1 * confidence_loss
        
        total_loss.backward()
        optimizer.step()
        
        train_losses.append(total_loss.item())
        
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred_coords_eval, pred_conf_eval = model(data['x'], data['edge_index'])
                
                # Evaluate only on hidden nodes
                hidden_pred = pred_coords_eval[data['hidden_mask']]
                hidden_true = data['target_coords'][data['hidden_mask']]
                
                mse = torch.mean((hidden_pred - hidden_true)**2).item()
                mae = torch.mean(torch.abs(hidden_pred - hidden_true)).item()
                
                print(f"Epoch {epoch}")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  Coord Loss: {coord_loss.item():.4f}")
                print(f"  Hidden Nodes MSE: {mse:.4f}")
                print(f"  Hidden Nodes MAE: {mae:.4f}")
                print(f"  Avg Confidence (visible): {pred_conf_eval[data['visible_mask']].mean():.3f}")
                print(f"  Avg Confidence (hidden): {pred_conf_eval[data['hidden_mask']].mean():.3f}")
                print("-" * 50)
            
            model.train()
    
    return model, train_losses

def evaluate_coordinate_prediction(model, data, coordinates):
    """
    Evaluate the coordinate prediction model
    """
    model.eval()
    with torch.no_grad():
        pred_coords, pred_confidence = model(data['x'], data['edge_index'])
        
        # Focus on hidden nodes (the actual prediction task)
        hidden_pred = pred_coords[data['hidden_mask']].numpy()
        hidden_true = data['target_coords'][data['hidden_mask']].numpy()
        
        # Compute metrics
        mse = mean_squared_error(hidden_true, hidden_pred)
        mae = mean_absolute_error(hidden_true, hidden_pred)
        
        # Distance errors
        distances = np.linalg.norm(hidden_pred - hidden_true, axis=1)
        mean_distance_error = np.mean(distances)
        median_distance_error = np.median(distances)
        
        # Spatial correlation
        corr_x = np.corrcoef(hidden_pred[:, 0], hidden_true[:, 0])[0, 1]
        corr_y = np.corrcoef(hidden_pred[:, 1], hidden_true[:, 1])[0, 1]
        
        return {
            'mse': mse,
            'mae': mae,
            'mean_distance_error': mean_distance_error,
            'median_distance_error': median_distance_error,
            'correlation_x': corr_x,
            'correlation_y': corr_y,
            'predictions': hidden_pred,
            'ground_truth': hidden_true,
            'confidence_scores': pred_confidence[data['hidden_mask']].numpy()
        }

def visualize_predictions(coordinates, data, model, output_path='coordinate_predictions.png'):
    """
    Visualize the coordinate predictions
    """
    model.eval()
    with torch.no_grad():
        pred_coords, pred_confidence = model(data['x'], data['edge_index'])
    
    pred_coords = pred_coords.numpy()
    pred_confidence = pred_confidence.numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Ground truth with visible/hidden distinction
    ax1 = axes[0]
    ax1.scatter(coordinates[data['visible_indices'], 0], 
               coordinates[data['visible_indices'], 1], 
               c='blue', s=20, alpha=0.7, label='Visible (Training)')
    ax1.scatter(coordinates[data['hidden_indices'], 0], 
               coordinates[data['hidden_indices'], 1], 
               c='red', s=20, alpha=0.7, label='Hidden (Target)')
    ax1.set_title('Ground Truth: Visible vs Hidden Nodes')
    ax1.legend()
    ax1.set_xlabel('X coordinate (μm)')
    ax1.set_ylabel('Y coordinate (μm)')
    
    # Plot 2: Predictions vs ground truth for hidden nodes
    ax2 = axes[1]
    ax2.scatter(coordinates[data['visible_indices'], 0], 
               coordinates[data['visible_indices'], 1], 
               c='lightblue', s=10, alpha=0.5, label='Visible')
    ax2.scatter(coordinates[data['hidden_indices'], 0], 
               coordinates[data['hidden_indices'], 1], 
               c='red', s=30, alpha=0.7, label='True Hidden')
    ax2.scatter(pred_coords[data['hidden_indices'], 0], 
               pred_coords[data['hidden_indices'], 1], 
               c='orange', s=30, alpha=0.7, marker='^', label='Predicted Hidden')
    ax2.set_title('Predictions vs Ground Truth')
    ax2.legend()
    ax2.set_xlabel('X coordinate (μm)')
    ax2.set_ylabel('Y coordinate (μm)')
    
    # Plot 3: Prediction confidence
    ax3 = axes[2]
    scatter = ax3.scatter(pred_coords[:, 0], pred_coords[:, 1], 
                         c=pred_confidence, s=20, cmap='viridis', alpha=0.7)
    ax3.set_title('Prediction Confidence')
    plt.colorbar(scatter, ax=ax3, label='Confidence Score')
    ax3.set_xlabel('X coordinate (μm)')
    ax3.set_ylabel('Y coordinate (μm)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig
