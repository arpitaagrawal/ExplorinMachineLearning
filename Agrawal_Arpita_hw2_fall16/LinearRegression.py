import sklearn.datasets as dt
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import operator
import itertools


boston_dataset = dt.load_boston()

index = 7


mtype = 'S10, i4'
correlation_vector = dict()

train_data = 'Training_Data'
train_target_data = 'Training_Data_Target'
test_data = 'Testing_Data'
test_target_data = 'Testing_Data_Target'
feature_names_label = 'feature_names'

def estimate_regression_theta(dataArray, targetDataArray):
    #print 'data array shape', np.shape(dataArray)
    num_test_examples = np.shape(dataArray)[0]
    column_of_ones = np.ones((num_test_examples,), dtype=np.int)
    joint_data_array =  np.column_stack((column_of_ones,dataArray))
    theta = np.dot(np.transpose(joint_data_array), joint_data_array)
    data_inv = np.linalg.pinv(np.matrix(theta))
    data_inv = np.dot(data_inv, np.transpose(joint_data_array))
    final_theta = np.dot(data_inv, targetDataArray)
    final_theta = np.array(final_theta).flatten()
    return final_theta

       
def normalizeParams(data_set_dict):
    mean = data_set_dict[train_data].mean(0)    
    std_1 = data_set_dict[train_data].std(0)

    for i in range(0,np.shape(data_set_dict[train_data])[0]):
        for j in range(0,np.shape(data_set_dict[train_data])[1]):
            data_set_dict[train_data][i][j] = (data_set_dict[train_data][i][j] - mean[j])/std_1[j]
            
            
    for i in range(0,np.shape(data_set_dict[test_data])[0]):
        for j in range(0,np.shape(data_set_dict[test_data])[1]):
            data_set_dict[test_data][i][j] = (data_set_dict[test_data][i][j] - mean[j])/std_1[j]
    return data_set_dict       


def seperate_train_test_data():
    
    boston = dt.load_boston().data
    target_data = dt.load_boston().target
    feature_names = dt.load_boston().feature_names
    
    boston_training_data = np.delete(boston, np.s_[::index], axis=0)
    boston_testing_data = boston[0::index].copy()
    boston_training_data_target = np.delete(target_data, np.s_[::index], axis=0)
    boston_testing_data_target = target_data[0::index].copy()
    
    #My main data set dictionary
    data_set_dict = dict()
    data_set_dict[train_data] = boston_training_data
    data_set_dict[train_target_data] = boston_training_data_target
    data_set_dict[test_data] = boston_testing_data
    data_set_dict[test_target_data] = boston_testing_data_target
    data_set_dict[feature_names_label] = feature_names
    return data_set_dict


def plotFeatureSubplots(data_set_dict, size):
    
    global correlation_vector
    plt.figure(1)
    plt.subplots_adjust(hspace=1)
    print 'Pearson cofficient of Feature \n'

    for i in range(0, np.shape(data_set_dict[feature_names_label])[0]):
        feature_subplot = plt.subplot(4, 4, i + 1)
        feature_subplot.set_title(data_set_dict[feature_names_label][i]) 
        plt.hist(data_set_dict[train_data][:, i], bins=size)
        r = find_correlation_vector(data_set_dict[train_data][:, i], data_set_dict[train_target_data])
        print 'Pearson cofficient of feature ', data_set_dict[feature_names_label][i], ' ::', r
        #r2 = np.correlate(data_set_dict[train_data][:, i], data_set_dict[train_target_data])
        correlation_vector[data_set_dict[feature_names_label][i]] = abs(r)
    print '\n\n'
    plt.show()
    #return feature

def find_correlation_vector(vector1, vector2):
    r = np.corrcoef(vector1, vector2)[0, 1]
    return r

def run_LR_with_all_features(theta2, dataArray, targetDataArray):
    
    num_test_examples = np.shape(dataArray)[0]
    column_of_ones = np.ones((num_test_examples), dtype=np.int)
    joint_data_array =  np.column_stack((column_of_ones,dataArray))
    estimated_value = np.dot(joint_data_array, np.transpose(theta2))
     
    residue_array = np.subtract(estimated_value, targetDataArray)
    return residue_array

def calculate_MSE(residue_array):
    num_test_examples = np.shape(residue_array)[0]
    train_data_MLE = float(1/(float(num_test_examples))) * np.sum(residue_array ** 2)
    #print train_data_MLE
    return train_data_MLE

def build_top_4_featureArray(feature_indexes_to_be_considered,):
    feature_arr= []
    feature_arr.append(feature_indexes_to_be_considered[0][0])
    feature_arr.append(feature_indexes_to_be_considered[1][0])
    feature_arr.append(feature_indexes_to_be_considered[2][0])
    feature_arr.append(feature_indexes_to_be_considered[3][0])
    return feature_arr
    
def run_LR_with_top_4_features(data_set_dict):
    print('\n\nStrategy 1: Select the 4 highest correlated features and then train the  linear regressor')
    global correlation_vector
    correlation_vector = sorted(correlation_vector.items(), key=operator.itemgetter(1), reverse=True)
    featr_arr = data_set_dict[feature_names_label]

    feature_indexes_to_be_considered = np.argwhere(np.logical_or(np.logical_or(featr_arr==correlation_vector[0][0], featr_arr==correlation_vector[1][0]), 
    np.logical_or(featr_arr==correlation_vector[2][0], featr_arr==correlation_vector[3][0]))) 
    
    feature_arr = build_top_4_featureArray(feature_indexes_to_be_considered)
   
    for i in feature_arr: print data_set_dict[feature_names_label][i]
    feature_indexes_to_be_considered = feature_arr
    theta = estimate_regression_theta(data_set_dict[train_data][:,feature_indexes_to_be_considered], data_set_dict[train_target_data])
    residue_array = run_LR_with_all_features(theta, data_set_dict[train_data][:,feature_indexes_to_be_considered], data_set_dict[train_target_data])   
    
    train_data_MLE = calculate_MSE(residue_array)
    print '\nMSE for Linear Regression with 4 highest correlated features (Training Data):::', train_data_MLE
    
    residue_array = run_LR_with_all_features(theta, data_set_dict[test_data][:,feature_indexes_to_be_considered], data_set_dict[test_target_data])   
    train_data_MLE = calculate_MSE(residue_array)
    print 'MSE for Linear Regression with 4 highest correlated features (Testing Data):::', train_data_MLE

def run_LR_with_iterative_features(data_set_dict):
    print('\n\nStrategy 2: Select the 4 features iteratively and then train the  linear regressor')
    global correlation_vector
    featr_arr = data_set_dict[feature_names_label]

    feature_indexes_to_be_considered = np.argwhere(featr_arr==correlation_vector[0][0]) 
    feature_arr= []
    feature_arr.append(feature_indexes_to_be_considered[0][0])

    feature_indexes_to_be_considered = feature_arr
    theta = estimate_regression_theta(data_set_dict[train_data][:,feature_indexes_to_be_considered], data_set_dict[train_target_data])
    residue_array = run_LR_with_all_features(theta, data_set_dict[train_data][:,feature_indexes_to_be_considered], data_set_dict[train_target_data])
    #Step find correlation between the residue array and the other feature vectors
    
    while(len(feature_arr) != 4):
        r_old = 0
        new_feature = -1
        for i in range(0, len(data_set_dict[feature_names_label])):
            if (i not in feature_arr):
                r = abs(find_correlation_vector(data_set_dict[train_data][:,i], residue_array))
                if r>r_old:
                    new_feature = i
                    r_old = r
                    
        feature_arr.append(new_feature)
        
        feature_indexes_to_be_considered = feature_arr
        theta = estimate_regression_theta(data_set_dict[train_data][:,feature_indexes_to_be_considered], data_set_dict[train_target_data])
        residue_array = run_LR_with_all_features(theta, data_set_dict[train_data][:,feature_indexes_to_be_considered], data_set_dict[train_target_data])
    
    for i in feature_arr: print data_set_dict[feature_names_label][i]
    train_data_MLE = calculate_MSE(residue_array)
    print '\nMSE for Linear Regression with 4 features calculated iteratively (Training Data):::', train_data_MLE
    residue_array = run_LR_with_all_features(theta, data_set_dict[test_data][:,feature_indexes_to_be_considered], data_set_dict[test_target_data])   
    train_data_MLE = calculate_MSE(residue_array)  
    print 'MSE for Linear Regression with 4 features calculated iteratively (Testing Data):::', train_data_MLE
   
def run_LR_with_4_best_features(data_set_dict):   
    print('\n\nStrategy 3: Brute force test all combinations of 4 features, find the best combo and then train the linear regressor')
    features_indices = range(0,13)
    for i in xrange(4,5):
        feature_cmbo_list = list(itertools.combinations(features_indices,i))
    
    old_cost = 9999999999
    best_theta = []
    best_feature_combo =[]

    for f in feature_cmbo_list:
        feature_arr= f
        
        feature_indexes_to_be_considered = feature_arr
        theta = estimate_regression_theta(data_set_dict[train_data][:,feature_indexes_to_be_considered], data_set_dict[train_target_data])
        
        residue_array = run_LR_with_all_features(theta, data_set_dict[train_data][:,feature_indexes_to_be_considered], data_set_dict[train_target_data])
        train_data_MLE = calculate_MSE(residue_array)
        if old_cost>train_data_MLE:
            best_feature_combo = feature_arr
            best_theta= theta
            old_cost = train_data_MLE
            
    for i in best_feature_combo: print data_set_dict[feature_names_label][i]        
    residue_array = run_LR_with_all_features(best_theta, data_set_dict[train_data][:,best_feature_combo], data_set_dict[train_target_data])
    train_data_MLE = calculate_MSE(residue_array)
    print 'MSE for Linear Regression with brute force best 4 features (Training Data):::', train_data_MLE
    
    
    residue_array = run_LR_with_all_features(best_theta, data_set_dict[test_data][:,best_feature_combo], data_set_dict[test_target_data])   
    train_data_MLE = calculate_MSE(residue_array) 
    print 'MSE for Linear Regression with brute force best 4 features (Testing Data):::', train_data_MLE
    
def build_new_feature_vector(data_set_dict):
    train_data_set = data_set_dict[train_data]
    test_data_set = data_set_dict[test_data]
    features_indices = range(0,13)
    for i in xrange(2,3):
        feature_cmbo_list = list(itertools.combinations_with_replacement(features_indices,i))
    #rint 'len::::', len(feature_cmbo_list), feature_cmbo_list
    for featr in feature_cmbo_list:
        #print featr[0]
        new_train_col = np.multiply(train_data_set[:,featr[0]], train_data_set[:,featr[1]])
        new_test_col = np.multiply(test_data_set[:,featr[0]], test_data_set[:,featr[1]])
        train_data_set = np.column_stack((train_data_set,new_train_col))
        test_data_set = np.column_stack((test_data_set,new_test_col))
        data_set_dict[train_data] = train_data_set
        data_set_dict[test_data] = test_data_set  
    data_set_dict = normalizeParams(data_set_dict)
    return data_set_dict

######################################################################

def run_LinearRegression():
    
    data_set_dict = seperate_train_test_data() 
    plotFeatureSubplots(data_set_dict, 10)   
    data_set_dict = normalizeParams(data_set_dict)
    theta = estimate_regression_theta(data_set_dict[train_data], data_set_dict[train_target_data]) 
    #print theta
    #MSE for training data
    residue_array = run_LR_with_all_features(theta, data_set_dict[train_data], data_set_dict[train_target_data])   
    train_data_MLE = calculate_MSE(residue_array)
    print 'MSE for Linear Regression with all the features (Training Data):::', train_data_MLE
    
    #MSE for testing data
    residue_array = run_LR_with_all_features(theta, data_set_dict[test_data], data_set_dict[test_target_data])   
    train_data_MLE = calculate_MSE(residue_array)
    print 'MSE for Linear Regression with all the features (Test Data):::', train_data_MLE

    run_LR_with_top_4_features(data_set_dict)
    run_LR_with_iterative_features(data_set_dict)
    run_LR_with_4_best_features(data_set_dict)

######################################################################################################           

def run_LR_polynomial_features():
    
    data_set_dict = seperate_train_test_data()
    print '\n\nPloynomial Expansion of features.\n'
    data_set_dict = build_new_feature_vector(data_set_dict)
    theta = estimate_regression_theta(data_set_dict[train_data], data_set_dict[train_target_data])
    residue_array = run_LR_with_all_features(theta, data_set_dict[train_data], data_set_dict[train_target_data])   
    train_data_MLE = calculate_MSE(residue_array)
    print 'MSE for Linear Regression with all the combinations of features (Training Data):::', train_data_MLE
     
    #MSE for testing data
    residue_array = run_LR_with_all_features(theta, data_set_dict[test_data], data_set_dict[test_target_data])   
    train_data_MLE = calculate_MSE(residue_array)
    print 'MSE for Linear Regression with all the combinations of (Test Data):::', train_data_MLE

