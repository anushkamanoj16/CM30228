import pandas as pd

# Load datasets
geolocation_df = pd.read_csv('./dataset/olist_geolocation_dataset.csv')
order_items_df = pd.read_csv('./dataset/olist_order_items_dataset.csv')
orders_df = pd.read_csv('./dataset/olist_orders_dataset.csv')
sellers_df = pd.read_csv('./dataset/olist_sellers_with_coordinates_dataset.csv')

# Average geolocation data by zipcode prefix
geolocation_avg = geolocation_df.groupby('geolocation_zip_code_prefix').agg({'geolocation_lat': 'mean', 'geolocation_lng': 'mean'}).reset_index()

# Load the customer dataset
customers_df = pd.read_csv('dataset/olist_customers_dataset.csv')

# Merge averaged geolocation data with customer information
customers_geo_avg = pd.merge(customers_df, geolocation_avg, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left').drop('geolocation_zip_code_prefix', axis=1)

# Merge orders with enriched customer data
orders_customers_geo = pd.merge(orders_df, customers_geo_avg, on='customer_id', how='left')

# Merge seller information in the combined dataset
orders_customers_sellers_geo = pd.merge(orders_customers_geo, order_items_df[['order_id', 'seller_id']], on='order_id', how='left')

# Merge seller coordinates based on seller ID
final_dataset = pd.merge(orders_customers_sellers_geo, sellers_df[['seller_id', 'seller_latitude', 'seller_longitude']], on='seller_id', how='left')

# Filter operations to clean the dataset
final_dataset_cleaned_missing = final_dataset.dropna(subset=['seller_id', 'geolocation_lat', 'geolocation_lng'])
final_dataset_delivered_only_nomissing = final_dataset_cleaned_missing[final_dataset_cleaned_missing['order_status'] == 'delivered']
final_dataset_nomissing = final_dataset_delivered_only_nomissing.dropna(subset=['order_delivered_customer_date'])

# Check  and remove duplicate entries
duplicate_rows = final_dataset_nomissing.duplicated(subset=['order_id', 'customer_id', 'seller_id', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'order_approved_at', 'order_purchase_timestamp', 'customer_unique_id', 'customer_zip_code_prefix'], keep=False)
final_model_dataset = final_dataset_nomissing[~duplicate_rows].reset_index(drop=True)

# Final dataset validation and inspection
print("Missing values in each column after all deletions:\n", final_model_dataset.isnull().sum())
print("\nShape of the final dataset without missing values or duplicates: ", final_model_dataset.shape)
print(final_model_dataset.head())

# Convert to .csv file
final_model_dataset.to_csv('final_model_dataset.csv', index=False)