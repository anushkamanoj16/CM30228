import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # Drop rows with NaN values
    data = data.dropna()
    features = data[['order_purchase_dayofweek', 'order_purchase_hour', 'order_purchase_season',
                     'delivery_duration', 'approval_duration', 'carrier_handling_duration', 'estimated_delivery_accuracy']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return data, features_scaled, scaler

def perform_pca(features_scaled, n_components=1):
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(features_scaled)
    return pca, pca_transformed

def train_and_evaluate_kmeans(data_pca, data, n_clusters=4, n_init=20, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    cluster_labels = kmeans.fit_predict(data_pca)
    score = silhouette_score(data_pca, cluster_labels)
    
    # Attach cluster IDs back to the original data (not the scaled features)
    data['tempcluster_id'] = cluster_labels
    
    return kmeans, score, data

if __name__ == "__main__":
    file_paths = ['./data/splits/tempclustering_train_dataset.csv', './data/splits/tempclustering_validation_dataset.csv', './data/splits/tempclustering_test_dataset.csv']
    for path in file_paths:
        data, features_scaled, scaler = load_and_preprocess_data(path)
        pca, data_pca = perform_pca(features_scaled)
        kmeans, score, data_with_clusters = train_and_evaluate_kmeans(data_pca, data)
        
        # Save modified dataset with cluster IDs and original data (including order_id)
        modified_data_path = path.replace('.csv', '_with_clusters.csv')
        data_with_clusters.to_csv(modified_data_path, index=False)
        
        print(f"{path} - Silhouette Score: {score}")

    # Save models to disk
    joblib.dump(scaler, 'models/temp_scaler.pkl')
    joblib.dump(pca, 'models/temp_pca.pkl')
    joblib.dump(kmeans, 'models/temp_kmeans.pkl')