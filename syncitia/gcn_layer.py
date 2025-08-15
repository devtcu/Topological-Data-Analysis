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

    def forward(self, x, edge_index, adj_powers):
        h = 0
        for n in range(self.max_order):
            norm_adj = torch_geometric.utils.to_dense_adj(
                torch_geometric.utils.dense_to_sparse(adj_powers[n])[0])[0]
            h += self.alpha[n] * self.conv(x, torch_geometric.utils.dense_to_sparse(norm_adj)[0])
        return h