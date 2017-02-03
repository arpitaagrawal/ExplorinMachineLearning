import random
import numpy as np
import matplotlib.pyplot as plt

is_debug = False
is_debug_2 = False
POLYNOMIAL_KERNEL = 'poly'

"""
 Arguments:
 k_arr : Array of Number of clusters to be executed for.
 Data : the Data which has to be clustered
 
 Returns:
 N*(K_arr length) array with each data point assigned to 
 a specific cluster acc. to the number of cluster argument KMean executed for 
 
"""

def kMeansCluster(k_arr, data, kernel, isPlot):
    
    for k in k_arr:    
        if is_debug:
            print 'kMeans Cluster being executed for k ', k
        prev_data = data
        if kernel == POLYNOMIAL_KERNEL:
            data = transform_data_using_kernel(data)
        print 'transformed data:::', data       
        rand_centroids = generate_k_rand_centroids(k, data)
        updated_centroids = rand_centroids
        prev_cluster_ids = np.empty([np.shape(data)[0]])
        count = 0
        while 1:
            #print 'count:', count
            count += 1
            cluster_ids = alot_clusters_to_points(data, updated_centroids)
            if np.array_equal(prev_cluster_ids, cluster_ids):
                break
            updated_centroids = update_centroids(data, cluster_ids, k)
            prev_cluster_ids = cluster_ids
            #print 'updated_centroids', updated_centroids
            
        #print cluster_ids
        if isPlot:
            plot_clusters(prev_data, cluster_ids, k)
    return  updated_centroids, cluster_ids

"""

Based on cluster IDs given to all the points calculates the updated centroids
By taking mean of all points in a given cluster 

"""         
def update_centroids(data, cluster_ids, k):
    updated_centroids = np.empty([k, np.shape(data)[1]])
    # Data point in consideration
    num_of_ele_in_k_cluster = np.zeros(k)
    for cur in range(0, np.shape(data)[0]):
        
        #k is one of the many clusters 
        k =  cluster_ids[cur]
        num_of_ele_in_k_cluster[k] = num_of_ele_in_k_cluster[k] + 1
        updated_centroids[k] = updated_centroids[k] + data[cur]
    
    for k in range(0,len(num_of_ele_in_k_cluster)):
        updated_centroids[k] = np.divide(updated_centroids[k],num_of_ele_in_k_cluster[k])
        
    return updated_centroids
    
"""

Assigns cluster ids to all data points based on the given centroids points provided.
Distance metric used : Euclidean Distanc : L2 Norm

"""  
def alot_clusters_to_points(data, centroids):
    
    dist_matrix = cal_dist_metric(data, centroids)
    #print dist_matrix
    cluster_ids = np.argmin(dist_matrix, axis=1)
    
    if is_debug_2:
            print 'Cluster ids for the given data set::', cluster_ids
    return cluster_ids


"""

Generates k random centroids for the initial run of our KMeans algorithm

"""        
def generate_k_rand_centroids(k, data):

    rand = random.sample(range(1, np.shape(data)[0]), k)
    i = 0
    rand_samples = np.empty([k, np.shape(data)[1]])
    for r in rand:
        rand_samples[i] = data[r]
        i += 1
        
    if is_debug:
        print 'Randomly generated Centroids::', rand_samples
    return rand_samples

"""

Calculates euclidean distance of all points in array a from each point in b

""" 
def cal_dist_metric(a, b):
    
    complete_dist_arr = []
    for i in range(np.shape(b)[0]):
        #print 'b:::', b[i]
        dist_arr = np.sqrt(np.sum((a-b[i])**2,axis=1))
        #print 'dist_arr',dist_arr
        if len(complete_dist_arr) == 0:
            complete_dist_arr = dist_arr
        else:
            complete_dist_arr = np.column_stack((complete_dist_arr,dist_arr))

    return complete_dist_arr


def plot_clusters(data, cluster_ids, k):
    
    N = np.shape(data)[0]
    x = data[:,0]
    y = data[:,1]
    colors = np.random.rand(k) * 3
    color_arr = np.zeros([N])
    for i in range(0, N):
        color_arr[i] = colors[cluster_ids[i]]
    area = np.pi * (4)**2  # 0 to 15 point radiuses
    
    plt.scatter(x, y, s=area, c=color_arr, alpha=0.5)
    plt.show()
    
"""

Transform the data of the form (x1, x2) into (x1, x2, x3) after applying the Kernel transformation: x_{1}^{2}, x_{1}^2 + x_{2}^2, x_{2}^{2}
Constraints : expects 2 dimensional data  and converts into 3 dimensional data set

"""     
def transform_data_using_kernel(data):
    
    transformed_feature_data_arr = np.empty([np.shape(data)[0], 3])
    count = 0
    for d in data:
        transformed_feature = np.empty([3])
        transformed_feature[0] = d[0] ** 2
        transformed_feature[1] = d[0] ** 2 + d[1] ** 2
        transformed_feature[2] = d[1] ** 2
        transformed_feature_data_arr[count] = transformed_feature
        count += 1
        
    return transformed_feature_data_arr