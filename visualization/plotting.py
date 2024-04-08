import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import googlemaps 
from datetime import datetime
from geopy.distance import great_circle

# # # Initialize the Google Maps client with my API key
# gmaps = googlemaps.Client(key='AIzaSyDKjnMtl-WlQroB2CIAh38TATcFOgxMzDo')

# # Function to fetch coordinates
# def fetch_coordinates(row):
#     # Formulate the query string
#     query = f"{row['seller_city']}, {row['seller_state']}, Brazil"
#     geocode_result = gmaps.geocode(query)
#     print("Fetching coordinates...")
#     if geocode_result:
#         lat = geocode_result[0]['geometry']['location']['lat']
#         lng = geocode_result[0]['geometry']['location']['lng']
#         return pd.Series([lat, lng], index=['latitude', 'longitude'])
#     else:
#         return pd.Series([None, None], index=['latitude', 'longitude'])


# # Apply the function to each row in your DataFrame
# sellers_df[['latitude', 'longitude']] = sellers_df.apply(fetch_coordinates, axis=1)

# print("savine to csv file now.")
# # Save the updated DataFrame to a new CSV file
# sellers_df.to_csv('olist_sellers_with_coordinates.csv', index=False)
# # print("done saving now.")

# Load datasets
orders_df = pd.read_csv('dataset/olist_orders_dataset.csv')
items_df = pd.read_csv('dataset/olist_order_items_dataset.csv')
sellers_df = pd.read_csv('dataset/olist_sellers_dataset.csv')
geolocation_df = pd.read_csv('dataset/olist_geolocation_dataset.csv')
customers_df = pd.read_csv('dataset/olist_customers_dataset.csv')
order_payments_df = pd.read_csv('dataset/olist_order_payments_dataset.csv')
sellers_with_coordinates_df = pd.read_csv('dataset/olist_sellers_with_coordinates.csv')


### Exploratory Data Analysis ###

# Summary Statistics and Dataset Structure
# print(orders_items_df.describe())
# print(orders_items_df.info())


# # # Plotting Monthly Order Volumes
# # orders_items_df.set_index('order_purchase_timestamp', inplace=True)
# # monthly_orders = orders_items_df.resample('ME').size()
# # plt.figure(figsize=(12, 6))
# # monthly_orders.plot(title='Monthly Order Volumes')
# # plt.xlabel('Month')
# # plt.ylabel('Number of Orders')
# # plt.show()
# # orders_items_df.reset_index(inplace=True)  # Reset index for further analysis

# # # Distribution of Delivery Times
# # plt.figure(figsize=(10, 6))
# # sns.histplot(orders_items_df['delivery_time'], bins=30, kde=True)
# # plt.title('Distribution of Delivery Times (in days)')
# # plt.xlabel('Delivery Time (days)')
# # plt.ylabel('Number of Orders')
# # plt.show()

# # # Hour of Day Distribution of Orders
# # # Directly create the 'hour_of_day' column from 'order_purchase_timestamp'
# # orders_items_df['hour_of_day'] = orders_items_df['order_purchase_timestamp'].dt.hour

# # plt.figure(figsize=(12, 6))
# # sns.countplot(x='hour_of_day', data=orders_items_df, palette='viridis')
# # plt.title('Distribution of Orders by Hour of Day')
# # plt.xlabel('Hour of Day')
# # plt.ylabel('Number of Orders')
# # plt.xticks(np.arange(0, 24, 1))
# # plt.grid(axis='y')
# # plt.show()

# # # Correlation Analysis
# # numeric_cols = orders_items_df.select_dtypes(include=[np.number])
# # corr_matrix = numeric_cols.corr()
# # plt.figure(figsize=(10, 8))
# # sns.heatmap(corr_matrix, annot=True, fmt=".2f")
# # plt.show()


# # # Histogram for delivery_time
# # plt.figure(figsize=(10, 6))
# # sns.histplot(orders_items_df['delivery_time'], bins=30, kde=True)
# # plt.title('Histogram of Delivery Time')
# # plt.xlabel('Delivery Time (normalized)')
# # plt.ylabel('Frequency')
# # plt.show()
# # # Box plot for delivery_time
# # plt.figure(figsize=(10, 6))
# # sns.boxplot(x=orders_items_df['delivery_time'])
# # plt.title('Box Plot of Delivery Time')
# # plt.xlabel('Delivery Time (normalized)')
# # plt.show()

# # # Histogram for freight_value
# # plt.figure(figsize=(10, 6))
# # sns.histplot(orders_items_df['freight_value'], bins=30, kde=True)
# # plt.title('Histogram of Freight Value')
# # plt.xlabel('Freight Value (normalized)')
# # plt.ylabel('Frequency')
# # plt.show()
# # # Box plot for freight_value
# # plt.figure(figsize=(10, 6))
# # sns.boxplot(x=orders_items_df['freight_value'])
# # plt.title('Box Plot of Freight Value')
# # plt.xlabel('Freight Value (normalized)')
# # plt.show()

# # # Histogram for payment_value
# # plt.figure(figsize=(10, 6))
# # sns.histplot(orders_items_df['payment_value'], bins=30, kde=True)
# # plt.title('Histogram of Payment Value')
# # plt.xlabel('Payment Value (normalized)')
# # plt.ylabel('Frequency')
# # plt.show()
# # # Box plot for payment_value
# # plt.figure(figsize=(10, 6))
# # sns.boxplot(x=orders_items_df['payment_value'])
# # plt.title('Box Plot of Payment Value')
# # plt.xlabel('Payment Value (normalized)')
# # plt.show()

# # # Histogram for price
# # plt.figure(figsize=(10, 6))
# # sns.histplot(orders_items_df['price'], bins=30, kde=True)
# # plt.title('Histogram of Price')
# # plt.xlabel('Price (normalized)')
# # plt.ylabel('Frequency')
# # plt.show()
# # # Box plot for price
# # plt.figure(figsize=(10, 6))
# # sns.boxplot(x=orders_items_df['price'])
# # plt.title('Box Plot of Price')
# # plt.xlabel('Price (normalized)')
# plt.show()

# # Check for missing values in the DataFrame
# missing_values = final_dataset.isnull().sum()
# print("Missing values in each column:\n", missing_values)

# # Check the percentage of missing values for each column
# missing_percentage = (final_dataset.isnull().sum() / len(final_dataset)) * 100
# print("\nPercentage of missing values in each column:\n", missing_percentage)
