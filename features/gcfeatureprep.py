import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import numpy as np
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data_path = "./data/final_model_dataset.csv"
df = pd.read_csv(data_path)

# Convert date columns to datetime
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
df['order_delivered_carrier_date'] = pd.to_datetime(df['order_delivered_carrier_date'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

# Calculate delivery timeframe in days
df['delivery_timeframe'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400

# Time of order
df['order_hour'] = df['order_purchase_timestamp'].dt.hour
df['order_weekday'] = df['order_purchase_timestamp'].dt.dayofweek

# Seasonality using your existing function
def get_season(dt):
    Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
    seasons = [
        (0, (date(Y, 1, 1), date(Y, 3, 20))),  # summer
        (1, (date(Y, 3, 21), date(Y, 6, 20))),  # autumn
        (2, (date(Y, 6, 21), date(Y, 9, 22))),  # winter
        (3, (date(Y, 9, 23), date(Y, 12, 20))),  # spring
        (0, (date(Y, 12, 21), date(Y, 12, 31)))  # summer again
    ]
    # Convert datetime to date for comparison if necessary
    if isinstance(dt, datetime):
        dt = dt.date()
    dt = dt.replace(year=Y)  # replace with dummy year
    return next(season for season, (start, end) in seasons if start <= dt <= end)

df['season'] = df['order_purchase_timestamp'].apply(get_season)

# Extract relevant geospatial features and calculate Haversine distance
geospatial_data = df[['geolocation_lat', 'geolocation_lng', 'seller_latitude', 'seller_longitude']].copy()
for col in ['geolocation_lat', 'geolocation_lng', 'seller_latitude', 'seller_longitude']:
    rad_col_name = col + '_rad'
    geospatial_data[rad_col_name] = np.radians(geospatial_data[col])

def calculate_distance(row):
    customer_coords = [row['geolocation_lat_rad'], row['geolocation_lng_rad']]
    seller_coords = [row['seller_latitude_rad'], row['seller_longitude_rad']]
    return haversine_distances([customer_coords, seller_coords])[0][1] * 6371  # Multiply by Earth's radius in km

geospatial_data['distance_km'] = geospatial_data.apply(calculate_distance, axis=1)
df['distance_km'] = geospatial_data['distance_km']

# Area Density
customer_density = df.groupby('customer_zip_code_prefix')['customer_id'].nunique().reset_index()
customer_density.rename(columns={'customer_id': 'customer_density'}, inplace=True)
df = pd.merge(df, customer_density, how='left', on='customer_zip_code_prefix')

# Normalize the features using StandardScaler
standard_features_to_normalize = ['delivery_timeframe', 'distance_km', 'customer_density']
standard_scaler = StandardScaler()
df[standard_features_to_normalize] = standard_scaler.fit_transform(df[standard_features_to_normalize])

# Save the standard scaler for future use
# joblib.dump(standard_scaler, './models/geoclusterscaler.pkl')

# If 'order_weekday' and 'order_hour' need a different treatment, use MinMaxScaler
minmax_features_to_normalize = ['order_hour', 'order_weekday']  # Adjust based on your requirement
minmax_scaler = MinMaxScaler()
df[minmax_features_to_normalize] = minmax_scaler.fit_transform(df[minmax_features_to_normalize])

# Save the MinMax scaler for future use
# joblib.dump(minmax_scaler, './models/geo_minmax_scaler.pkl')
# Save the DataFrame with the new features and normalized values to a new CSV file
new_data_path = "./data/final_gcmodel_dataset_with_features.csv"
df.to_csv(new_data_path, index=False)

# Assuming 'df' is your DataFrame after all preprocessing steps
# Define features for clustering
features = ['order_id','geolocation_lat', 'geolocation_lng', 'seller_latitude', 'seller_longitude', 'distance_km']

# Splitting the dataset into training, validation, and testing sets
X_train, X_temp = train_test_split(df[features], test_size=0.2, random_state=42)
X_validation, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

# Define the base path for saving the split datasets
base_path = './data/splits/'

# Save the datasets
X_train.to_csv(f'{base_path}geoclustering_train_dataset.csv', index=False)
X_validation.to_csv(f'{base_path}geoclustering_validation_dataset.csv', index=False)
X_test.to_csv(f'{base_path}geoclustering_test_dataset.csv', index=False)