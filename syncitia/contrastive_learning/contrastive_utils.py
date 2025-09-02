import torch
import numpy as np
import copy
from torch_geometric.utils import dropout_edge

def create_graph_views(data, edge_dropout_prob=0.7, feature_mask_prob=0.3):
    """
    Create two augmented views of the graph for contrastive learning
    
    Args:
        data: PyTorch Geometric data object
        edge_dropout_prob: Probability of dropping edges
        feature_mask_prob: Probability of masking features
        
    Returns:
        Two augmented versions of the graph data
    """
    # Create two views with the same structure but different augmentations
    view1 = copy.deepcopy(data)
    view2 = copy.deepcopy(data)
    
    # Edge dropout for view 1
    edge_index1, _ = dropout_edge(
        data.edge_index, p=edge_dropout_prob, force_undirected=True
    )
    view1.edge_index = edge_index1
    
    # Edge dropout for view 2
    edge_index2, _ = dropout_edge(
        data.edge_index, p=edge_dropout_prob, force_undirected=True
    )
    view2.edge_index = edge_index2
    
    # Feature masking for view 1
    feature_mask1 = torch.rand_like(data.x) > feature_mask_prob
    view1.x = data.x * feature_mask1
    
    # Feature masking for view 2
    feature_mask2 = torch.rand_like(data.x) > feature_mask_prob
    view2.x = data.x * feature_mask2
    
    return view1, view2

def train_contrastive_model(
    model,
    data,
    optimizer,
    criterion,
    num_epochs=200,
    lambda_contrast=0.1,
    edge_dropout_prob: float = 0.2,
    feature_mask_prob: float = 0.3,
    log_every: int = 10,
):
    """
    Train the coordinate prediction model with contrastive learning
    
    Args:
        model: ContrastiveGCN model
        data: PyTorch Geometric data object
        optimizer: PyTorch optimizer
        criterion: Loss function for coordinate prediction
        num_epochs: Number of training epochs
        lambda_contrast: Weight for contrastive loss term
    """
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Create two augmented views of the graph
        view1, view2 = create_graph_views(
            data,
            edge_dropout_prob=edge_dropout_prob,
            feature_mask_prob=feature_mask_prob,
        )
        
        # Forward pass with embeddings on both views
        pred_coords1, embeddings1 = model(view1, return_embeddings=True)
        pred_coords2, embeddings2 = model(view2, return_embeddings=True)
        
        # Coordinate prediction loss (only on hidden nodes)
        # Use original data for ground truth coordinates
        coord_loss = criterion(
            pred_coords1[data.hidden_mask], 
            data.target_coords[data.hidden_mask]
        )
        
        # Contrastive loss between the two views
        contrastive_loss = model.contrastive_loss(embeddings1, embeddings2)
        
        # Combined loss
        total_loss = coord_loss + lambda_contrast * contrastive_loss
        
        total_loss.backward()
        optimizer.step()
        
        train_losses.append(total_loss.item())

        # Evaluation during training
        if epoch % max(1, log_every) == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                pred_coords_eval = model(data)

                # Evaluate only on hidden nodes
                hidden_pred = pred_coords_eval[data.hidden_mask]
                hidden_true = data.target_coords[data.hidden_mask]

                mse = torch.mean((hidden_pred - hidden_true)**2).item()
                distances = torch.norm(hidden_pred - hidden_true, dim=1)
                mean_dist = torch.mean(distances).item()

                print(
                    f"Epoch {epoch:03d} | Total: {total_loss:.4f} | Coord: {coord_loss.item():.4f} | "
                    f"Contrast: {contrastive_loss.item():.4f} | MSE: {mse:.2f} | MeanDist: {mean_dist:.2f} μm"
                )

            model.train()
    
    return model, train_losses

def evaluate_contrastive_model(model, data):
    """
    Evaluate the contrastive model on coordinate prediction
    
    Args:
        model: ContrastiveGCN model
        data: PyTorch Geometric data object
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        pred_coords = model(data)
        
        # Evaluate only on hidden nodes
        hidden_pred = pred_coords[data.hidden_mask]
        hidden_true = data.target_coords[data.hidden_mask]
        
        # Calculate metrics
        mse = torch.mean((hidden_pred - hidden_true)**2).item()
        mae = torch.mean(torch.abs(hidden_pred - hidden_true)).item()
        
        distances = torch.norm(hidden_pred - hidden_true, dim=1)
        mean_distance = torch.mean(distances).item()
        median_distance = torch.median(distances).item()
        
        # Calculate correlations
        hidden_pred_np = hidden_pred.cpu().numpy()
        hidden_true_np = hidden_true.cpu().numpy()
        
        correlation_x = np.corrcoef(hidden_pred_np[:, 0], hidden_true_np[:, 0])[0, 1]
        correlation_y = np.corrcoef(hidden_pred_np[:, 1], hidden_true_np[:, 1])[0, 1]
        
        results = {
            'mse': mse,
            'mae': mae,
            'mean_distance_error': mean_distance,
            'median_distance_error': median_distance,
            'correlation_x': correlation_x,
            'correlation_y': correlation_y,
            'predictions': hidden_pred_np,
            'ground_truth': hidden_true_np
        }
        
        return results
