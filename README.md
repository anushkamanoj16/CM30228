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

## IDE Setup
Please ensure final_model_dataset.csv is moved to the ./data directory.
Please ensure there exists a directory calls "splits" within the data directory. 
Please ensure olist_sellers_with_coordinates.csv is moved to the ./dataset directory.
The run.sh script requires the following commands to be entered into you terminal in order to run:
chmod +x run.sh
./run.sh 

Please note, if you would not like to create a virtual environment to run the project please comment out line 8-12 in run.sh.
