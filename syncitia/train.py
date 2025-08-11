import torch
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

def train_model(model, data, optimizer, criterion, num_epochs=100, task='classification'):
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        if task == 'coordinate':
            # Simple coordinate prediction
            pred_coords = model(data)
            
            # Loss only on hidden nodes (the nodes we're trying to predict)
            loss = criterion(pred_coords[data.hidden_mask], 
                           data.target_coords[data.hidden_mask])
        else:
            # Original classification training
            out = model(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Evaluation during training
        if epoch % 20 == 0:
            if task == 'coordinate':
                model.eval()
                with torch.no_grad():
                    pred_coords_eval = model(data)
                    hidden_pred = pred_coords_eval[data.hidden_mask]
                    hidden_true = data.target_coords[data.hidden_mask]
                    mse = torch.mean((hidden_pred - hidden_true)**2).item()
                    distances = torch.norm(hidden_pred - hidden_true, dim=1)
                    mean_dist = torch.mean(distances).item()
                    
                    print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, MSE: {mse:.2f}, Mean Distance: {mean_dist:.2f} μm")
                model.train()
            else:
                # Original classification evaluation
                model.eval()
                with torch.no_grad():
                    pred = model(data).argmax(dim=1)
                    train_acc = accuracy_score(data.y[data.train_mask].cpu(), 
                                             pred[data.train_mask].cpu())
                    test_acc = accuracy_score(data.y[data.test_mask].cpu(), 
                                            pred[data.test_mask].cpu())
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
                model.train()
    
    return model

def evaluate_model(model, data, task='classification'):
    model.eval()
    with torch.no_grad():
        if task == 'coordinate':
            pred_coords, pred_confidence = model(data)
            
            # Focus on hidden nodes (the actual prediction task)
            hidden_pred = pred_coords[data.hidden_mask].numpy()
            hidden_true = data.target_coords[data.hidden_mask].numpy()
            
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
                'confidence_scores': pred_confidence[data.hidden_mask].numpy()
            }
        else:
            # Original classification evaluation
            out = model(data)
            predictions = out.argmax(dim=1).numpy()
            accuracy = accuracy_score(data.y.cpu(), predictions)
            return predictions, accuracy

def get_feature_importance(model):
    w1 = model.conv1.conv.lin.weight.detach().cpu().numpy()
    weight_importance = np.mean(np.abs(w1), axis=0)
    feature_names = ['Betti-0', 'Betti-1', 'Degree']
    return {name: weight_importance[i] for i, name in enumerate(feature_names)}