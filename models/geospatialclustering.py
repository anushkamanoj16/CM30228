import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import joblib

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    features = df[['geolocation_lat', 'geolocation_lng', 'seller_latitude', 'seller_longitude', 'distance_km']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return df, features_scaled, scaler

def train_and_evaluate_kmeans(features_scaled, df, n_clusters=2, n_init=10, max_iter=300, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, max_iter=max_iter, random_state=random_state)
    cluster_labels = kmeans.fit_predict(features_scaled)
    score = silhouette_score(features_scaled, cluster_labels)
    # Attach cluster IDs back to the original dataset 
    df['geocluster_id'] = cluster_labels
    
    return kmeans, score, df

if __name__ == "__main__":
    file_paths = ['./data/splits/geoclustering_train_dataset.csv', './data/splits/geoclustering_validation_dataset.csv', './data/splits/geoclustering_test_dataset.csv']
    for path in file_paths:
        df, features_scaled, scaler = load_and_preprocess_data(path)
        kmeans, silhouette_score_value, df_with_clusters = train_and_evaluate_kmeans(features_scaled, df)
        # Save modified dataset with cluster IDs and original data 
        modified_data_path = path.replace('.csv', '_with_clusters.csv')
        df_with_clusters.to_csv(modified_data_path, index=False)
        
        print(f"{path} - Silhouette Score: {silhouette_score_value}")

