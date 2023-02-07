import os
import sys
import torch
import torch_geometric
import torch.nn.functional as F
from torch_sparse import SparseTensor, set_diag
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from typing import Optional
from torch.nn import Parameter
from torch_geometric.utils import softmax

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import src.utils as utils


class ModelPretreatment:
    def __init__(self, add_self_loops: bool = True, normalize: bool = True):
        super(ModelPretreatment, self).__init__()
        self.add_self_loops = add_self_loops
        self.normalize = normalize

    def pretreatment(self, x: Tensor, edge_index: Adj):
        # add self loop
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x.size(0)
                edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
                edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # normalize
        edge_weight = None
        if self.normalize:
            if isinstance(edge_index, Tensor):
                row, col = edge_index
            elif isinstance(edge_index, SparseTensor):
                row, col, _ = edge_index.coo()
            deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return edge_index, edge_weight


# backbone model with DropMessage
class BbGCN(MessagePassing):
    def __init__(self, add_self_loops: bool = True, normalize: bool = True):
        super(BbGCN, self).__init__()
        self.pt = ModelPretreatment(add_self_loops, normalize)
        self.edge_weight = None

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        edge_index, self.edge_weight = self.pt.pretreatment(x, edge_index)
        y = self.propagate(edge_index=edge_index, size=None, x=x, drop_rate=drop_rate)
        return y

    def message(self, x_j: Tensor, drop_rate: float):
        # normalize
        if self.edge_weight is not None:
            x_j = x_j * self.edge_weight.view(-1, 1)

        if not self.training:
            return x_j

        # drop messages
        x_j = F.dropout(x_j, drop_rate)

        return x_j


class BbGAT(MessagePassing):
    def __init__(self, in_channels: int, heads: int = 1, add_self_loops: bool = True):
        super(BbGAT, self).__init__(node_dim=0)
        self.pt = ModelPretreatment(add_self_loops, False)
        self.heads = heads
        self.edge_weight = None

        # parameters
        self.att = Parameter(torch.Tensor(1, heads, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        utils.glorot(self.att)

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        x_l = x_r = x.view(-1, 1, x.size(-1)).repeat(1, self.heads, 1)
        alpha_l = alpha_r = (x_l * self.att).sum(dim=-1)

        edge_index, _ = self.pt.pretreatment(x, edge_index)

        y = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), drop_rate=drop_rate)
        y = y.view(-1, self.heads * x.size(-1))
        return y

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int], drop_rate: float):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)

        if not self.training:
            return x_j * alpha.unsqueeze(-1)

        # drop messages
        x_j = F.dropout(x_j, drop_rate)

        return x_j * alpha.unsqueeze(-1)


class BbAPPNP(MessagePassing):
    def __init__(self, K: int, alpha: float, add_self_loops: bool = True, normalize: bool = True):
        super(BbAPPNP, self).__init__()
        self.pt = ModelPretreatment(add_self_loops, normalize)
        self.K = K
        self.alpha = alpha
        self.edge_weight = None

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        edge_index, self.edge_weight = self.pt.pretreatment(x, edge_index)
        h = x
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, drop_rate=drop_rate)
            x = x * (1 - self.alpha)
            x += self.alpha * h
        return x

    def message(self, x_j: Tensor, drop_rate: float):
        # normalize
        if self.edge_weight is not None:
            x_j = x_j * self.edge_weight.view(-1, 1)

        if not self.training:
            return x_j

        # drop messages
        x_j = F.dropout(x_j, drop_rate)

        return x_j
