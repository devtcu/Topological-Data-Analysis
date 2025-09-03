import torch
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.nn import GCNConv

class CoordinatePredictionGCN(torch.nn.Module):
    """GCN for predicting missing nuclei coordinates"""
    
    def __init__(self, in_channels, hidden_channels=64):
        super(CoordinatePredictionGCN, self).__init__()
        
        #feature extraction layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels//2)
        
        #prediction head (x, y coordinates) - NOTE: this is something I can tune in the future
        self.coord_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels//2, hidden_channels//4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_channels//4, 2)  # (x, y) coordinates
        )
        
        #how certain are we?
        self.confidence_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels//2, hidden_channels//4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels//4, 1),
            torch.nn.Sigmoid()  # 0-1 confidence score
        )
        
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x, edge_index):
        # Feature extraction through graph convolutions
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = self.dropout(h1)
        
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = self.dropout(h2)
        
        h3 = F.relu(self.conv3(h2, edge_index))
        
        # lets see the predictions
        coordinates = self.coord_head(h3)  # Shape: [N, 2]
        confidence = self.confidence_head(h3).squeeze()  # Shape: [N]
        
        return coordinates, confidence
