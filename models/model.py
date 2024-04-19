import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class HydrogelGNNPINN(nn.Module):
    def __init__(self, gnn_input_dim, gnn_hidden_dim, gnn_output_dim, pinn_input_dim, pinn_hidden_dim, pinn_output_dim):
        super(HydrogelGNNPINN, self).__init__()
        
        # GNN layers
        self.gnn_conv1 = gnn.GCNConv(gnn_input_dim, gnn_hidden_dim)
        self.gnn_conv2 = gnn.GCNConv(gnn_hidden_dim, gnn_output_dim)
        
        # PINN layers
        self.pinn_fc1 = nn.Linear(pinn_input_dim, pinn_hidden_dim)
        self.pinn_fc2 = nn.Linear(pinn_hidden_dim, pinn_output_dim)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, data):
        # GNN forward pass
        x, edge_index = data.x, data.edge_index
        x = self.gnn_conv1(x, edge_index)
        x = self.relu(x)
        x = self.gnn_conv2(x, edge_index)
        gnn_output = x
        
        # PINN forward pass
        pinn_input = torch.cat([gnn_output, data.x], dim=1)
        x = self.pinn_fc1(pinn_input)
        x = self.relu(x)
        pinn_output = self.pinn_fc2(x)
        
        return pinn_output