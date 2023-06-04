from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class DeProp_Prop(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, lambda1: float, lambda2: float, gamma: float,
                 improved: bool = False,  cached: bool = True,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        super(DeProp_Prop, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma = gamma
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)
        z = (1 - self.gamma * self.lambda1 + self.gamma * self.lambda2) * x
        s = self.gamma * self.lambda1 * self.propagate(edge_index, x=x, edge_weight=edge_weight,
                              size=None)
        t = self.gamma * self.lambda2 * x @ (x.t() @ x)
        out = z + s - t

        if self.bias is not None:
            out += self.bias

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


    def __repr__(self):
        return '{}(lambda1={}, lambda2={}, gamma={})'.format(
            self.__class__.__name__, self.lambda1, self.lambda2, self.gamma)








