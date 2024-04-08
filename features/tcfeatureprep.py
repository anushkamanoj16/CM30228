import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from datetime import date, datetime
from sklearn.model_selection import train_test_split

# Load the dataset
final_model_dataset = pd.read_csv('./data/final_model_dataset.csv')

# Convert date columns to datetime
final_model_dataset['order_purchase_timestamp'] = pd.to_datetime(final_model_dataset['order_purchase_timestamp'])
final_model_dataset['order_approved_at'] = pd.to_datetime(final_model_dataset['order_approved_at'])
final_model_dataset['order_delivered_carrier_date'] = pd.to_datetime(final_model_dataset['order_delivered_carrier_date'])
final_model_dataset['order_delivered_customer_date'] = pd.to_datetime(final_model_dataset['order_delivered_customer_date'])
final_model_dataset['order_estimated_delivery_date'] = pd.to_datetime(final_model_dataset['order_estimated_delivery_date'])

# Create new features for durations
final_model_dataset['delivery_duration'] = ((final_model_dataset['order_delivered_customer_date'] - final_model_dataset['order_purchase_timestamp']).dt.total_seconds())/3600
final_model_dataset['approval_duration'] = ((final_model_dataset['order_approved_at'] - final_model_dataset['order_purchase_timestamp']).dt.total_seconds())/3600
final_model_dataset['carrier_handling_duration'] = ((final_model_dataset['order_delivered_carrier_date'] - final_model_dataset['order_approved_at']).dt.total_seconds())/3600
final_model_dataset['estimated_delivery_accuracy'] = ((final_model_dataset['order_estimated_delivery_date'] - final_model_dataset['order_delivered_customer_date']).dt.total_seconds())/3600

# Day of the week and time of the day
final_model_dataset['order_purchase_dayofweek'] = final_model_dataset['order_purchase_timestamp'].dt.dayofweek
final_model_dataset['order_purchase_hour'] = final_model_dataset['order_purchase_timestamp'].dt.hour

def get_season(dt):
    Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
    seasons = [
        (0, (date(Y, 1, 1), date(Y, 3, 20))),  # summer
        (1, (date(Y, 3, 21), date(Y, 6, 20))),  # autumn
        (2, (date(Y, 6, 21), date(Y, 9, 22))),  # winter
        (3, (date(Y, 9, 23), date(Y, 12, 20))),  # spring
        (0, (date(Y, 12, 21), date(Y, 12, 31)))  # summer again
    ]
    
    # Ensure dt is a date object for comparison
    dt = dt.date() if isinstance(dt, datetime) else dt
    dt = dt.replace(year=Y)  # replace with dummy year
    
    return next(season for season, (start, end) in seasons if start <= dt <= end)

# Apply the get_season function to your dataset
final_model_dataset['order_purchase_season'] = final_model_dataset['order_purchase_timestamp'].apply(get_season)

# Initialize the scaler for normalising the newly created columns
scaler = StandardScaler()

# List of the duration features we want to scale
duration_features = ['delivery_duration', 'approval_duration', 'carrier_handling_duration', 'estimated_delivery_accuracy']

# Fit the scaler on the duration features and transform the data
final_model_dataset[duration_features] = scaler.fit_transform(final_model_dataset[duration_features])

# Normalize additional features if needed
# Example for day of the week and hour:
minmax_scaler = MinMaxScaler()
final_model_dataset[['order_purchase_dayofweek', 'order_purchase_hour']] = minmax_scaler.fit_transform(final_model_dataset[['order_purchase_dayofweek', 'order_purchase_hour']])

# Save the scaler to a file
#joblib.dump(scaler, './models/tempclusterscaler.pkl')

# Save the updated dataset with the new scaled duration features
final_model_dataset.to_csv('./data/final_model_dataset_with_features.csv', index=False) # Currently hashed as data has already been created.

# Load the newly created dataset 
final_model_dataset = pd.read_csv('./data/final_model_dataset_with_features.csv')

# Selecting only the relevant features for clustering
features = final_model_dataset[['order_id','order_purchase_dayofweek', 'order_purchase_hour', 'order_purchase_season',
                                'delivery_duration', 'approval_duration', 'carrier_handling_duration', 'estimated_delivery_accuracy']]

# Splitting the dataset into training, validation, and testing sets
X_train, X_temp = train_test_split(features, test_size=0.2, random_state=42)
X_validation, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

base_path = './data/splits/'

# Save the datasets
X_train.to_csv(f'{base_path}tempclustering_train_dataset.csv', index=False)
X_validation.to_csv(f'{base_path}tempclustering_alidation_dataset.csv', index=False)
X_test.to_csv(f'{base_path}tempclustering_test_dataset.csv', index=False)