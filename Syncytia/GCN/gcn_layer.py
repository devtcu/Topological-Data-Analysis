#change this file if you dont need it
import torch
import torch_geometric.utils
from torch_geometric.nn import GCNConv

class HigherOrderGCNLayer(torch.nn.Module): 
    def __init__(self, in_channels, out_channels, max_order=2):
        super(HigherOrderGCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.alpha = torch.nn.Parameter(torch.ones(max_order))
        self.max_order = max_order

    def forward(self, x, edge_index, adj_powers=None):
        """
        Forward pass for higher-order GCN layer.

        If adj_powers is not provided, compute adjacency powers from edge_index
        up to self.max_order.
        """
        # Compute adjacency powers if not provided
        if adj_powers is None:
            adj = torch_geometric.utils.to_dense_adj(edge_index)[0]
            adj_powers = [adj]
            for _ in range(1, self.max_order):
                adj_powers.append(torch.matmul(adj_powers[-1], adj))

        h = 0
        for n in range(self.max_order):
            # Convert dense adjacency to edge_index for GCNConv
            edge_index_n = torch_geometric.utils.dense_to_sparse(adj_powers[n])[0]
            h = h + self.alpha[n] * self.conv(x, edge_index_n)
        return h