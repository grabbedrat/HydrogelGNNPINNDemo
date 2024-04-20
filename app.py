import os
from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from models.model import HydrogelGNNPINN
from utils.data_utils import create_graph_data
from utils.evaluation_utils import evaluate_model
from data.data_generation import generate_random_data
from utils.visualization_utils import plot_drug_release
from config import *

app = Flask(__name__)

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

@app.route('/')
def index():
    # Generate default example data
    default_hydrogel_properties = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    default_drug_release_profiles = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]]
    
    return render_template('index.html', default_hydrogel_properties=default_hydrogel_properties, default_drug_release_profiles=default_drug_release_profiles)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    hydrogel_properties = request.json['hydrogel_properties']
    drug_release_profiles = request.json['drug_release_profiles']
    
    # Create graph data
    data = create_graph_data(hydrogel_properties, drug_release_profiles)
    
    # Load the trained model
    model_path = os.path.join(app.root_path, 'trained_model.pth')
    model = HydrogelGNNPINN(GNN_INPUT_DIM, GNN_HIDDEN_DIM, GNN_OUTPUT_DIM, PINN_INPUT_DIM, PINN_HIDDEN_DIM, PINN_OUTPUT_DIM)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Make predictions
    predicted_release = model(data).detach().numpy()[0]
    
    # Prepare the response
    response = {
        'predicted_release': predicted_release.tolist()
    }
    
    return jsonify(response)

if __name__ == '__main__':
    # Generate random data for demonstration
    hydrogel_properties, drug_release_profiles = generate_random_data(num_samples=100)
    
    # Create graph data
    data = create_graph_data(hydrogel_properties, drug_release_profiles)
    
    # Train the model
    model = train_model([data])
    
    # Save the trained model weights
    model_path = os.path.join(app.root_path, 'trained_model.pth')
    torch.save(model.state_dict(), model_path)
    
    app.run(debug=True)