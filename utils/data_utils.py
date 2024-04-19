import torch
from torch_geometric.data import Data

def create_graph_data(hydrogel_properties, drug_release_profiles):
    # Create a Data object for graph-based input
    x = torch.tensor(hydrogel_properties, dtype=torch.float)
    y = torch.tensor(drug_release_profiles, dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data