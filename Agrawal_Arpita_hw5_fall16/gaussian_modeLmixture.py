import numpy as np
import random
import math
import matplotlib.pyplot as plt
import kMeans
from scipy.stats import multivariate_normal

is_debug = False
is_debug_2 = False
num_of_iterations = 500

mu_arr = []
sigma_arr = []
phi_arr = []


def cluster_using_GMM(data, k):
    print "\nClustring data using Gausian Mixture Model"
    log_prob_arr = np.empty([5, num_of_iterations])
    for num_of_runs in range(0,5):
        initialize_mean_covar(data, k)
        [prob_density, log_prob] = run_expectation_maxim(data, k)
        cluster_id = np.argmax(prob_density, axis =1)
        #print cluster_id
        plot_clusters(data, cluster_id, k)
        log_prob_arr[num_of_runs] = log_prob
        print 'Final Mean for run ',num_of_runs,', =:::', mu_arr
        print 'Final Sigma for run ',num_of_runs,', =:::', sigma_arr
        print 'Final Phi Arr for run ',num_of_runs,', =:::', phi_arr
    #print log_prob_arr[0]
    #print log_prob_arr[1]
    plot_log(log_prob_arr)

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
    
def run_expectation_maxim(data, k):
    
    global mu_arr
    
    log_prob = np.zeros([num_of_iterations])
    for iter in range(0, num_of_iterations):
        prev_mu_arr= np.copy(mu_arr)
        ## E Step
        prob_density = run_expectation_step(data, k)
        
        ## M Step
        run_maximization_step(prob_density, data, k)
        log_prob[iter] = calculate_log_likelihood(prob_density)
        if np.array_equal(mu_arr, prev_mu_arr):
            break
        print 'Iteration Number ::', iter
    #plot_log(log_prob)
    return prob_density, log_prob

def plot_log(log_prob):
    
    plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow', 'black', 'magenta'])
    mylist = list(xrange(len(log_prob[0])))
    plt.plot(mylist, log_prob[0])
    plt.plot(mylist, log_prob[1])
    plt.plot(mylist, log_prob[2])
    plt.plot(mylist, log_prob[3])
    plt.show()
    
def calculate_log_likelihood(prob_density):
    print "Prob densityyy:"
    log_prob_sum = 0
    for i in range(0, np.shape(prob_density)[0]):
        sum_prob = np.sum(prob_density[i])
        log_prob = math.log(sum_prob)
        log_prob_sum = log_prob_sum + log_prob
    return log_prob_sum       
"""
Expectation Step :: Calculates the probability of each point being in each cluster j/ each gaussian distribution j

w_{j}^{i} = (g_{j}(x) phi(j))/ (/sum_{l=1}^{k} g_{l}(x)phi_{l})
where w_{j}^{i} : is the probability of the ith training example to be in the cluster j

Returns: Matrix of size N X K --> each row being probability of a data point 'i' in being each cluster 'j'
"""    
def run_expectation_step(data, k):
    print '\n\nrunning expectation step'
    
    gaus_density = np.empty([np.shape(data)[0], k])
    for cluster in range(0,k):
        # calculate probability of each point being in cluster  
        gaus_density[:, cluster] = gaus_prob_value(data, cluster) * phi_arr[cluster]
    
    # for each data in the dataset:
    for i in range(0, np.shape(data)[0]):
        #probability of each data point being in the cluster j
        for j in range(0,k):
            gaus_density[i,j] = gaus_density[i,j]/(np.sum(gaus_density[i,]))    
    #print 'np.shape(prob density of each point w.r.t each gaussian)', np.shape(gaus_density) 
    return gaus_density
 
def run_maximization_step(prob_density, data, k):
    print '\n\n running maximization step'
    
    # Calculate updated phi array based on the points alotted to each cluster
    phi_arr = np.mean(prob_density, axis = 0)
    if is_debug:
        print 'Updated phi array after maximization step::', phi_arr
    update_mean_arr(prob_density, data, k)
    update_variance_arr(prob_density, data, k)

"""
 Updating the mean array based on the weights obtained in the E step
 multiply the weight of each training example with each cluster with x_{i} / sum(w_{ij}*mu_{j})
"""
def update_mean_arr(prob_density, data, k):
    
    if is_debug:
        print 'updating mean array'
    global mu_arr
    
    """
        prob_density is N X K matrix, probability of each point N w.r.t to each cluster K
        data is N X D matrix, where N is the number of data points and D is the number of features in each data point
        mu --> should be a 1XD vector i.e the same dimension as a single data point in our training set
        we have to assign mean value to each cluster j
    """    
    
    for j in range(0,k):
        tot_sum_of_prob = np.sum(prob_density[:, j])
        weighted_sum = np.dot([prob_density[:, j]], data)
        mu_arr[j] =  weighted_sum/ tot_sum_of_prob
        
def update_variance_arr(prob_density, data, k):
    
    global sigma_arr
    if is_debug:
        print 'updating sigma array'
    for j in range(0,k):
        tot_sum_of_prob = np.sum(prob_density[:, j], axis = 0)
        
        data_mu_diff = data - mu_arr[j]
        sigma_k = np.zeros([np.shape(data)[1], np.shape(data)[1]])
        for i in range(np.shape(data)[0]):
            prod = np.dot(np.transpose([data_mu_diff[i]]), [data_mu_diff[i]])
            sigma_k = sigma_k + np.multiply(prob_density[i][j], prod)
        sigma_arr[j] = sigma_k / tot_sum_of_prob
        
        
"""
tested working fine. Compared with sklearn libraries
"""                  
def gaus_prob_value(data, cluster):
    
    global mu_arr, sigma_arr, phi_arr
    num_of_features = 2
    #gaus_prob = np.empty(np.shape(data)[0])
    if is_debug_2:
        print 'Calculating gaussian probability for cluster ::', cluster
    pdf = np.empty([np.shape(data)[0]])
    i = 0
    for d in data:
        data_mean_diff = d - mu_arr[cluster]
        
#         if is_debug:
#             print 'data mean diff arr::', data_mean_diff
         
        pi_term = pow(2*(np.pi), num_of_features)
        #print 'np.shape(sigma_arr[cluster]):::', np.shape(sigma_arr[cluster]), sigma_arr[cluster] 
        #print "sigma_arr[cluster]", sigma_arr[cluster]
        norm_const = (1 /float(np.sqrt(pi_term * np.linalg.det(sigma_arr[cluster]))))
        interm =  np.dot(data_mean_diff, np.linalg.pinv(sigma_arr[cluster]))
        
        #print np.dot(interm,np.transpose(data_mean_diff))
        pdf[i] = norm_const * np.exp(-0.5 * np.dot(interm,np.transpose(data_mean_diff)))
        #y = multivariate_normal.pdf(d, mean=mu_arr[cluster], cov=sigma_arr[cluster]);
        #print '\n\ny:::',y
        #print '\n\npdf:::',pdf[i]
        i+=1
    return pdf
 
 
###################################################################
   
"""
    Tested Seems to be work ing fine
"""        
def pick_k_rand_points(data, k):
    rand = random.sample(range(1, np.shape(data)[0]), k)
    i = 0
    rand_samples = np.empty([k, np.shape(data)[1]])
    for r in rand:
        rand_samples[i] = data[r]
        i += 1
    
    if is_debug:    
        print 'Randomly generated mean points::', rand_samples
    return rand_samples

def pick_k_rand_mean(data, k):
    global mu_arr
    [mu_arr, cluster_ids] = kMeans.kMeansCluster([k], data, "", False)
    return [mu_arr, cluster_ids]
     
def initialize_k_covar(data, k):
    global sigma_arr
    covar = np.cov(np.transpose(data)) 
    print 'data shape', np.shape(data)
    print 'shape', np.shape(covar)
    print "determinant of covar matrix", np.linalg.det(covar)
    
    sigma_arr = np.empty([k, np.shape(covar)[0], np.shape(covar)[1]])
    for i in range(0,k):
        sigma_arr[i] =covar
    return sigma_arr

def intialize_phi_arr(data, k):
    phi_arr = np.empty([k])
    num_of_trainin_examples = np.shape(data)[0]
    eq_no_of_samples = float(num_of_trainin_examples)/float(k)
    for i in range(0,k):
        phi_arr[i] = eq_no_of_samples/float(num_of_trainin_examples)
    return phi_arr

"""
    Tested Seems to be work ing fine
"""
def initialize_phi_using_kMeans(k, cluster_ids):
    
    global phi_arr
    
    phi_arr = np.empty([k])
    count = np.zeros([k])
    for c in cluster_ids:
        count[c] = count[c] + 1

    for i in range(0,k):
        phi_arr[i] = float(count[i])/ float(len(cluster_ids))
        
    #print 'Newly initialised phi array :::', phi_arr
    
"""
    Tested Seems to be work ing fine
"""    
def initialize_k_covar_using_kMeans(data, k, cluster_ids):
    global mu_arr, sigma_arr
    sigma_arr = np.empty([k, np.shape(data)[1], np.shape(data)[1]])
    for j in range(0,k):
        part_data = data[np.where( cluster_ids == j )]
        part_data = part_data - mu_arr[j]
        #print np.shape(np.transpose(part_data))
        #print np.shape(part_data)
        sigma_arr[j] = (np.dot(np.transpose(part_data), part_data))/float(np.shape(part_data)[0] -1)
        #print sigma_arr[j]
"""
    This function intialises the initial mean and variances randomly
    Can also use kMeans to intialise the data-- future enhancement
    mu : set to random data points from data set
    covar: set to covariance of the entire data set
    phi: set to equal probability of each cluster i.e gaussian distribution
"""    
def initialize_mean_covar(data, k):
    print "\nRandom Initializing of params in progress"
    
    global mu_arr, sigma_arr, phi_arr
    mu_arr = pick_k_rand_points(data, k)
    #[mu_arr, cluster_ids] = pick_k_rand_mean(data, k)
    #initialize_phi_using_kMeans(k, cluster_ids)
    #initialize_k_covar_using_kMeans(data, k, cluster_ids)
    phi_arr = intialize_phi_arr(data, k)
    #pick_k_rand_mean(data, k)
    initialize_k_covar(data, k)
    
    
    if is_debug:
        print "Initial mu_arr ::", mu_arr
        print "\n\nInitial sigma_arr :::", sigma_arr
        print "\n\nInitial phi_arr::::", phi_arr