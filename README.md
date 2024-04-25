# Urban Logistics Optimization Project

## Overview
This project is aimed at optimizing urban logistics through the use of clustering and reinforcement learning techniques. It involves preprocessing data, feature engineering, clustering temporal and geospatial data, and finally implementing a custom reinforcement learning environment for route optimization.

## Setting Up a Virtual Environment
To create and activate a virtual environment, follow these steps:

### For macOS and Linux:
python3 -m venv venv
source venv/bin/activate

## Prerequisites
Before running the scripts, ensure that you have Python 3.x installed along with the following packages:

- pandas
- numpy
- scikit-learn
- joblib
- gym
- stable-baselines3
- matplotlib
- seaborn

You can install them using the following command:
pip3 install pandas numpy scikit-learn joblib gym stable-baselines3 matplotlib seaborn

## The data preparation step involves cleaning and structuring raw data into a suitable format for further processing. Run the following script to start the data preparation:
python3 ./data/dataprep.py

## To enhance the model's performance, we engineer temporal and geospatial features from the data. Run the following scripts in order to prepare features for clustering:
python3 ./features/tcfeatureprep.py
python3 ./features/gcfeatureprep.py

## Clustering is used as a pre-processing step to simplify the reinforcement learning problem. Run the following scripts to perform clustering:
python3 ./models/temporalclustering.py
python3 ./models/geospatialclustering.py

## After clustering, run the following script to prepare features for the reinforcement learning model:
python3 ./features/rlfeatureprep.py

## The final step is to run the reinforcement learning model, which will use the preprocessed and clustered data to optimize delivery routes:
python3 ./models/reinforcementlearning.py


## To run the scriptis in run.sh please write the following command in your terminal:
Run the command chmod +x setup_project.sh in your terminal. This makes the script executable.
You can run the script by typing ./setup_project.sh in your terminal. Make sure you are in the directory where the script is located.