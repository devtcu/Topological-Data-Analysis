import torch
import torch.nn.functional as F
import torch_geometric.utils
from gcn_layer import HigherOrderGCNLayer

class ContrastiveGCN(torch.nn.Module):
    """GCN with contrastive learning for nuclei coordinate prediction"""
    
    def __init__(self, in_channels, hidden_channels=64, proj_dim=32, tau=0.5):
        super(ContrastiveGCN, self).__init__()
        
        # Feature extraction layers using higher-order GCN
        self.conv1 = HigherOrderGCNLayer(in_channels, hidden_channels, max_order=2)
        self.conv2 = HigherOrderGCNLayer(hidden_channels, hidden_channels//2, max_order=2)
        
        # Projection head for contrastive learning
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels//2, proj_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_dim, proj_dim)
        )
        
        # Coordinate prediction head
        self.coord_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels//2, hidden_channels//4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels//4, 2)  # (x, y) coordinates
        )
        
        self.dropout = torch.nn.Dropout(0.2)
        self.tau = tau  # Temperature parameter for contrastive loss
    
    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index
        
        # Feature extraction through graph convolutions
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = self.dropout(h1)
        
        h2 = F.relu(self.conv2(h1, edge_index))
        
        # Node embeddings for contrastive learning
        z = self.proj(h2)
        
        # Coordinate prediction
        coordinates = self.coord_head(h2)  # Shape: [N, 2]
        
        if return_embeddings:
            return coordinates, z
        return coordinates
    
    def contrastive_loss(self, z1, z2):
        """
        Compute contrastive loss between two views of the graph
        
        Args:
            z1, z2: Node embeddings from two graph views
        
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings to unit length
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.t()) / self.tau
        
        # Labels are on the diagonal (positive pairs)
        labels = torch.arange(sim_matrix.size(0), device=z1.device)
        
        # InfoNCE loss (cross-entropy with identity matrix as target)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
