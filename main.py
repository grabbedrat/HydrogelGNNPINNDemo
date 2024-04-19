import torch
import numpy as np
from models.model import HydrogelGNNPINN
from utils.data_utils import create_graph_data
from utils.evaluation_utils import evaluate_model
from data.data_generation import generate_random_data
from utils.visualization_utils import plot_drug_release
from config import *

def train_model(data_loader):
    model = HydrogelGNNPINN(GNN_INPUT_DIM, GNN_HIDDEN_DIM, GNN_OUTPUT_DIM, PINN_INPUT_DIM, PINN_HIDDEN_DIM, PINN_OUTPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = torch.mean((output - data.y) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}")
    
    return model

def evaluate_model_performance(model, data_loader):
    loss = evaluate_model(model, data_loader)
    print(f"Evaluation Loss: {loss:.4f}")

if __name__ == "__main__":
    # Generate random data for demonstration
    hydrogel_properties, drug_release_profiles = generate_random_data(num_samples=100)
    
    # Create graph data
    data = create_graph_data(hydrogel_properties, drug_release_profiles)
    
    # Train the model
    model = train_model([data])
    
     # Evaluate the model
    evaluate_model_performance(model, [data])
    
    # Plot and save the drug release graph
    time_steps = np.arange(drug_release_profiles.shape[1])
    predicted_release = model(data).detach().numpy()[0]
    actual_release = data.y.detach().numpy()[0]
    plot_drug_release(time_steps, predicted_release, actual_release, "drug_release_graph.png")