Cancer_SGNN

Cancer_SGNN uses Graph Neural Networks (GNNs) to perform survival analysis for cancer patients with Whole Slide Imaging (WSI) data, aiming to assist clinical decision - making.
Technical Details
Graph & Explanation: Graph construction and GNNExplainer adopt SGMF implementation. SGMF extracts graph structures from WSI data for GNN input, and GNNExplainer interprets model predictions.
Data: Put constructed graph data and patient survival data in the data folder after graph construction.
Usage
Environment Setup
Navigate to the environment.yml directory in the terminal.
Create a Conda environment: conda env create -f environment.yml.
Activate the environment: conda activate <env_name> (replace <env_name> with the actual name).
Run the Model
Execute main.py to start training and testing. The script will load data, train the GNN model, and test its performance in predicting patient survival.
