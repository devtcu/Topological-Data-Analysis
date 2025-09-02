import torch
import torch.nn.functional as F
import torch_geometric.utils
from gcn_layer import HigherOrderGCNLayer

class HigherOrderGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, max_order=2, task='classification'):
        super(HigherOrderGCN, self).__init__()
        self.task = task
        self.conv1 = HigherOrderGCNLayer(in_channels, hidden_channels, max_order)
        self.conv2 = HigherOrderGCNLayer(hidden_channels, out_channels, max_order)
        self.dropout = torch.nn.Dropout(0.1)
        
        # Task-specific heads
        if task == 'coordinate':
            # Simple coordinate prediction: final layer -> 2D coordinates
            self.coord_head = torch.nn.Linear(out_channels, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        adj = torch_geometric.utils.to_dense_adj(edge_index)[0]
        adj_powers = [adj]
        for _ in range(1, self.conv1.max_order):
            adj_powers.append(torch.matmul(adj_powers[-1], adj))
        
        h = self.conv1(x, edge_index, adj_powers)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, adj_powers)
        h = F.relu(h)
        
        if self.task == 'coordinate':
            # Just predict coordinates - no confidence
            return self.coord_head(h)
        else:
            # Original classification task
            return h