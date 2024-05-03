import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full datasets
geo_full = pd.read_csv('./data/final_gcmodel_dataset_with_features.csv')  
temp_full = pd.read_csv('./data/final_model_dataset_with_features.csv')  

geo_train_with_clusters = pd.read_csv('./data/splits/geoclustering_train_dataset_with_clusters.csv')
geo_validation_with_clusters = pd.read_csv('./data/splits/geoclustering_validation_dataset_with_clusters.csv')
geo_test_with_clusters = pd.read_csv('./data/splits/geoclustering_test_dataset_with_clusters.csv')

temp_train_with_clusters = pd.read_csv('./data/splits/tempclustering_train_dataset_with_clusters.csv')
temp_validation_with_clusters = pd.read_csv('./data/splits/tempclustering_validation_dataset_with_clusters.csv')
temp_test_with_clusters = pd.read_csv('./data/splits/tempclustering_test_dataset_with_clusters.csv')

geo_full_with_clusters = pd.concat([geo_train_with_clusters, geo_validation_with_clusters, geo_test_with_clusters], ignore_index=True)
temp_full_with_clusters = pd.concat([temp_train_with_clusters, temp_validation_with_clusters, temp_test_with_clusters], ignore_index=True)

# Merge the full datasets on 'order_id'
merged_full = pd.merge(geo_full, temp_full, on='order_id', suffixes=('_geo', '_temp'))

clusters_full = pd.merge(geo_full_with_clusters[['order_id', 'geocluster_id']], temp_full_with_clusters[['order_id', 'tempcluster_id']], on='order_id')

final_merged_dataset = pd.merge(merged_full, clusters_full, on='order_id')

final_merged_dataset.to_csv('./data/rl_merged_dataset.csv', index=False)

# Load the final merged datasets
final_merged = pd.read_csv('./data/rl_merged_dataset.csv')  

# Define a list of relevant features
relevant_features = [
    'order_id',
    'geolocation_lat_geo', 'geolocation_lng_geo',
    'seller_latitude_geo', 'seller_longitude_geo',
    'distance_km',
    'geocluster_id',
    'customer_state_geo',
    'order_purchase_timestamp_geo',
    'estimated_delivery_accuracy', 'delivery_duration', 'order_estimated_delivery_date_geo', 'approval_duration', 'carrier_handling_duration',
    'tempcluster_id', 
    'customer_density',
    'order_delivered_customer_date_geo',
]

# Select the relevant features from the final merged dataset
final_rl_dataset = final_merged[relevant_features]

# Check for NaN values in the final_rl_dataset
nan_counts = final_rl_dataset.isna().sum()

# Print a report of NaN values for each column
print("NaN values report for each column:")
print(nan_counts)

# Print columns with at least one NaN value
columns_with_nan = nan_counts[nan_counts > 0].index.tolist()
if columns_with_nan:
    print("\nColumns with at least one NaN value:")
    print(columns_with_nan)
else:
    print("\nNo NaN values found.")

# Save the trimmed dataset for the RL model
final_rl_dataset.to_csv('./data/rl_final_model_dataset.csv', index=False)
