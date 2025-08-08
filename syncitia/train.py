import torch
import numpy as np
from sklearn.metrics import accuracy_score

def train_model(model, data, optimizer, criterion, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                train_acc = accuracy_score(data.y[data.train_mask].cpu(), 
                                         pred[data.train_mask].cpu())
                test_acc = accuracy_score(data.y[data.test_mask].cpu(), 
                                        pred[data.test_mask].cpu())
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            model.train()
    return model

def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        predictions = out.argmax(dim=1).numpy()
        accuracy = accuracy_score(data.y.cpu(), predictions)
    return predictions, accuracy

def get_feature_importance(model):
    w1 = model.conv1.conv.lin.weight.detach().cpu().numpy()
    weight_importance = np.mean(np.abs(w1), axis=0)
    feature_names = ['Betti-0', 'Betti-1', 'Degree']
    return {name: weight_importance[i] for i, name in enumerate(feature_names)}