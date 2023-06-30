import numpy as np
import torch
import torch.nn as nn
from torch_cluster import knn_graph, radius_graph
from torch.nn import Sequential, Linear, ReLU, ModuleList
from torch_geometric.nn import MessagePassing, MetaLayer 
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class EdgeModel_ini(torch.nn.Module):
    def __init__(self,node_in, node_out, edge_in, edge_out, hid_channels):
        super().__init__()
        layers = [Linear(edge_in, hid_channels),
                  ReLU(),
                  Linear(hid_channels, edge_out)]
        self.edge_mlp = Sequential(*layers)

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        return self.edge_mlp(edge_attr)

class EdgeModel(torch.nn.Module):
    def __init__(self,node_in, node_out, edge_in, edge_out, hid_channels):
        super().__init__()
        layers = [Linear(node_in*2 + edge_in, hid_channels),
                  ReLU(),
                  Linear(hid_channels, edge_out)]
        self.edge_mlp = Sequential(*layers)

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self,node_in, node_out, edge_in, edge_out, hid_channels): 
        super().__init__()        
        self.node_mlp = Sequential(Linear(node_in+3*edge_out,hid_channels),
                                   ReLU(), 
                                   Linear(hid_channels, node_out))

    def forward(self, x, edge_index, edge_attr,u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = edge_attr 
        
        out1 = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out2 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out3 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out1, out2, out3], dim=1)

        return self.node_mlp(out)


class NodeModel_ini(torch.nn.Module):
    def __init__(self,node_in, node_out, edge_in, edge_out, hid_channels): 
        # node_in: number features of x
        # size_out: 
        super().__init__()
        
        self.node_mlp = Sequential(Linear(3*edge_out,hid_channels),
                                   ReLU(), 
                                   Linear(hid_channels, node_out))

    def forward(self, x, edge_index, edge_attr,u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = edge_attr
        
        out1 = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out2 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out3 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([out1, out2, out3], dim=1)

        return self.node_mlp(out)


class GNN(torch.nn.Module):
  def __init__(self,trial, n_layers,dim_out,node_in,hid_channels,edge_in):
    super().__init__()

    self.n_layers = n_layers
    self.dim_out = dim_out
    edge_in = edge_in
    node_out = hid_channels
    edge_out = hid_channels
    
    layers = []
    # Encoder graph block
    inlayer = MetaLayer(node_model=NodeModel_ini(node_in, node_out, edge_in, edge_out, hid_channels),
                        edge_model=EdgeModel_ini(node_in, node_out, edge_in, edge_out, hid_channels))
    layers.append(inlayer)
    # Change input node and edge feature sizes
    node_in = node_out
    edge_in = edge_out

    for i in range(n_layers-1):
      op = MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels),
                     edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels))
      layers.append(op)
    self.layers = ModuleList(layers)
        
    self.outlayer = Sequential(Linear(node_out*3, hid_channels),
                              ReLU(),
                              Linear(hid_channels, hid_channels),
                              ReLU(),
                              Linear(hid_channels, hid_channels),
                              ReLU(),
                              Linear(hid_channels, self.dim_out))

  def forward(self,data):
    x,edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

    # Message passing layers
    for layer in self.layers:
      x, edge_attr, __ = layer(x, edge_index, edge_attr, batch=data.batch) #x, edge_index, edge_attr,u, batch

    # Multipooling layer
    addpool = global_add_pool(x, batch)
    meanpool = global_mean_pool(x, batch) # [n_graph, F_x]
    maxpool = global_max_pool(x, batch)

    out = torch.cat([addpool,meanpool,maxpool], dim=1)

    out = self.outlayer(out) # [n_graph, 3*node_out]
    return out
