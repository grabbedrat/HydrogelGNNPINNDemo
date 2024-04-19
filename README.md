# Hydrogel Drug Release Demo with GNNs and PINNs

This demo showcases the application of Graph Neural Networks (GNNs) and Physics-Informed Neural Networks (PINNs) for modeling hydrogel drug release. The demo uses random sample data to demonstrate the implementation and training of the models.

## File Structure

- `data/`: Contains scripts and utilities for generating and handling random sample data.
  - `random_sample_data.py`: Script to generate random sample data for hydrogel properties and drug release profiles.
  - `README.md`: Provides information about the data generation process and any assumptions made.

- `models/`: Contains the implementation of the GNN and PINN models.
  - `gnn_model.py`: Defines the architecture and forward pass of the GNN model.
  - `pinn_model.py`: Defines the architecture and forward pass of the PINN model.
  - `README.md`: Provides an overview of the model architectures and any specific considerations.

- `utils/`: Contains utility scripts for data preprocessing, plotting, and other helper functions.
  - `data_utils.py`: Utility functions for data loading, preprocessing, and formatting.
  - `plotting_utils.py`: Utility functions for visualizing the results and generating plots.
  - `README.md`: Provides information about the utility functions and their usage.

- `notebooks/`: Contains Jupyter notebooks for data exploration, model training, and evaluation.
  - `data_exploration.ipynb`: Notebook for exploring and visualizing the random sample data.
  - `model_training.ipynb`: Notebook for training the GNN and PINN models using the random sample data.
  - `model_evaluation.ipynb`: Notebook for evaluating the trained models and visualizing the results.

- `requirements.txt`: Lists the required Python packages and their versions for running the demo.

- `README.md`: Provides an overview of the demo, installation instructions, and usage guidelines.

- `main.py`: The main script to run the demo, including data generation, model training, and evaluation.

## Installation

1. Clone the repository:

git clone https://github.com/your-repo/hydrogel-drug-release-demo.git


2. Navigate to the project directory:

cd hydrogel-drug-release-demo


3. Install the required packages:

pip install -r requirements.txt


## Usage

1. Generate random sample data:

python data/random_sample_data.py


2. Train the GNN and PINN models:

python main.py --train


3. Evaluate the trained models:

python main.py --evaluate


4. Explore the notebooks in the `notebooks/` directory for more detailed analysis and visualizations.

## License

This project is licensed under the [MIT License](LICENSE).