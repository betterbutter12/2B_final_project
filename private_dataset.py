import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering 
import numpy as np


data = pd.read_csv('private_data.csv')
features = data[['1', '2', '3', '4', '5', '6']]


for i in range(1, 6):
    for j in range(i + 1, 7):
        plt.figure(figsize=(8, 8))
        plt.scatter(data[str(i)], data[str(j)], s=0.1)
        plt.xlabel(f'S{i}')
        plt.ylabel(f'S{j}')
        plt.title(f'S{i} vs S{j}')
        plt.savefig(f'private_graphs/s{i}_s{j}.png')
plt.close('all') 

# Data preprocessing: Standardization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA (You can still use PCA for dimensionality reduction before clustering)
# Consider experimenting with n_components if needed.
pca = PCA(n_components=0.9)  # Retain 90% variance
reduced_features = pca.fit_transform(scaled_features)
print(f"Number of PCA components: {reduced_features.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")


print("Applying HDBSCAN for initial clustering...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=750,  
    min_samples=25,     
    gen_min_span_tree=True,
    metric='euclidean'  
)
labels = clusterer.fit_predict(reduced_features)

# Ensure 4*6-1 clusters (post-process if needed)
n_clusters_required = 4 * 6 - 1
n_hdbscan_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"HDBSCAN initially found {n_hdbscan_clusters} clusters (excluding noise).")

if n_hdbscan_clusters < n_clusters_required:
    print(f"Less than {n_clusters_required} clusters from HDBSCAN. ")
    mask = labels == -1 
    noise_points = reduced_features[mask]

    if np.sum(mask) > 0: # If there are noise points
        num_clusters_for_noise = n_clusters_required - n_hdbscan_clusters
        if num_clusters_for_noise > 0:
            print(f"Applying Spectral Clustering to {np.sum(mask)} noise points to form {num_clusters_for_noise} additional clusters.")
            spectral_noise_clusterer = SpectralClustering(
                n_clusters=num_clusters_for_noise,
                affinity='nearest_neighbors',
                n_neighbors=min(10, len(noise_points) - 1 if len(noise_points) > 1 else 1), 
                assign_labels='kmeans',
                random_state=42,
                n_init=10
            )
            labels_noise = spectral_noise_clusterer.fit_predict(noise_points)
            labels[mask] = labels_noise + n_hdbscan_clusters
        else:
            print("No need to apply Spectral Clustering to noise.")
    else:
        print("No noise points (-1) from HDBSCAN, so no Spectral Clustering applied.")
elif n_hdbscan_clusters > n_clusters_required:
    print(f"HDBSCAN found more than {n_clusters_required} clusters.")

output = pd.DataFrame({'id': data['id'], 'label': labels})
output.to_csv('grading/private_submission.csv', index=False)

