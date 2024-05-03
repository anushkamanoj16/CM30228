#!/bin/bash

# Urban Logistics Optimization Project Setup Script

# Exit script if any command fails
set -e

# Create and activate a virtual environment
echo "Creating a virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "Virtual environment activated."

# Install required Python packages
echo "Installing required packages..."
pip3 install pandas numpy scikit-learn joblib gym stable-baselines3 matplotlib seaborn tensorboard shimmy
echo "Packages installed."

# Data preparation
echo "Starting data preparation..."
python3 ./data/dataprep.py
echo "Data preparation complete."

# Feature engineering for temporal and geospatial data
echo "Preparing temporal and geospatial features..."
python3 ./features/tcfeatureprep.py
python3 ./features/gcfeatureprep.py
echo "Features prepared."

# Clustering
echo "Performing clustering..."
python3 ./models/temporalclustering.py
python3 ./models/geospatialclustering.py
echo "Clustering complete."

# Reinforcement learning feature preparation
echo "Preparing features for reinforcement learning..."
python3 ./features/rlfeatureprep.py
echo "RL features ready."

# Run the reinforcement learning model
echo "Running the reinforcement learning model..."
python3 ./models/reinforcementlearning.py
echo "Reinforcement learning model execution complete."

echo "Setup and model execution complete."

