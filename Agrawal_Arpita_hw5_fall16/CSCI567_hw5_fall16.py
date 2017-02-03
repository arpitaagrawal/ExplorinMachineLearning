import read_data
from kMeans import kMeansCluster
from gaussian_modeLmixture import cluster_using_GMM
from gaussian_modeLmixture import plot_clusters
read_data.read_data()

# # Executing kMeans on blob dataset with k =2,3,5
kMeansCluster([2,3,5], read_data.blob_train_data, "", True)
# 
# # Executing kMeans on rinh shaped dataset with k =2,3,5
kMeansCluster([2,3,5], read_data.circle_train_data, "", True)
# 
# # Executing Kernel kMeans on ring shaped dataset with k =2 and polynomial transformation applied to the dataset
kMeansCluster([2], read_data.circle_train_data, "poly", True)

#plot_clusters(read_data.blob_train_data, [0,1], 3)
cluster_using_GMM(read_data.blob_train_data, 3)
