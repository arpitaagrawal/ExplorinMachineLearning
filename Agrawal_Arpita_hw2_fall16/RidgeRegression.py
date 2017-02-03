from numpy.linalg import inv
import numpy as np
import sklearn.datasets as dt


train_data = 'Training_Data'
train_target_data = 'Training_Data_Target'
test_data = 'Testing_Data'
test_target_data = 'Testing_Data_Target'
feature_names_label = 'feature_names'
    
boston_training_data_target = []
boston_testing_data_target = []
boston_training_data = []
boston_testing_data = []
  
def load_data():

    global boston_testing_data_target, boston_training_data_target, boston_testing_data, boston_training_data
    
    boston = dt.load_boston().data
    target_data = dt.load_boston().target
    
       
    index = 7
    boston_testing_data = boston[0::index].copy()
    boston_training_data = np.delete(boston, np.s_[::index], axis=0)
    
    boston_training_data_target = np.delete(target_data, np.s_[::index], axis=0)
    boston_testing_data_target = target_data[0::index].copy()
    
    mean = boston_training_data.mean(0)    
    std_1 = boston_training_data.std(0)

    for i in range(0,np.shape(boston_training_data)[0]):
        for j in range(0,np.shape(boston_training_data)[1]):
            boston_training_data[i][j] = (boston_training_data[i][j] - mean[j])/std_1[j]
                
                
    for i in range(0,np.shape(boston_testing_data)[0]):
        for j in range(0,np.shape(boston_testing_data)[1]):
            boston_testing_data[i][j] = (boston_testing_data[i][j] - mean[j])/std_1[j]


def ridge_estimator(dataArray, targetDataArray, lamda):
    num_test_examples = np.shape(dataArray)[0]
    feature_length = np.shape(dataArray)[1]
    
    column_of_ones = np.ones((num_test_examples,), dtype=np.int)
    joint_data_array =  np.column_stack((column_of_ones,dataArray))
    
    identity_lmda_product = lamda * np.identity(feature_length + 1)
    temp = np.dot(np.transpose(joint_data_array), joint_data_array)

    summation_term = np.add(temp, identity_lmda_product)
    data_inv = np.dot(np.linalg.pinv(summation_term), np.transpose(joint_data_array))
    final_theta = np.dot(data_inv, targetDataArray)
    return final_theta 

#, 5.4001, 2.4001, 10, 0.8401, 0.6401

def predict_value(theta2, dataArray, targetDataArray):
    
    num_test_examples = np.shape(dataArray)[0]
    column_of_ones = np.ones((num_test_examples), dtype=np.int)
    joint_data_array =  np.column_stack((column_of_ones,dataArray))
    estimated_value = np.dot(joint_data_array, np.transpose(theta2))
     
    residue_array = np.subtract(estimated_value, targetDataArray)
    return residue_array


def calculate_MSE(residue_array):
    num_test_examples = np.shape(residue_array)[0]
    train_data_MSE = float(1/(float(num_test_examples))) * np.sum(residue_array ** 2)
    return train_data_MSE

def run_ridge_estimator():
    
    load_data()
    global boston_training_data, boston_training_data_target, boston_testing_data, boston_testing_data_target
    print '\n\nRidge Regression for 7th data point is test sample strategy\n'
    for lamda in [0.01,0.1,1.0]:
        print 'Lamda = ', lamda
        final_theta = ridge_estimator(boston_training_data, boston_training_data_target, lamda)
        residue_array = predict_value(final_theta, boston_training_data, boston_training_data_target)
        print 'MSE for Training Data :::',calculate_MSE(residue_array)
        
        residue_array = predict_value(final_theta, boston_testing_data, boston_testing_data_target)
        print 'MSE for Testing Data :::',calculate_MSE(residue_array)
        print '\n' 
        
    print 'Ridge Regression for Cross Validation strategy\n' 

def normalize_data(data_set_dict):
    mean = data_set_dict[train_data].mean(0)    
    std_1 = data_set_dict[train_data].std(0)

    for i in range(0,np.shape(data_set_dict[train_data])[0]):
        for j in range(0,np.shape(data_set_dict[train_data])[1]):
            data_set_dict[train_data][i][j] = (data_set_dict[train_data][i][j] - mean[j])/std_1[j]
            
            
    for i in range(0,np.shape(data_set_dict[test_data])[0]):
        for j in range(0,np.shape(data_set_dict[test_data])[1]):
            data_set_dict[test_data][i][j] = (data_set_dict[test_data][i][j] - mean[j])/std_1[j]
    return data_set_dict


def ridge_estimator_CV():
    
    load_data()
    global boston_training_data, boston_training_data_target, boston_testing_data, boston_testing_data_target
    old_lamda = -10
    old_MSE = 10000000  
    joint_data_array =  np.column_stack((boston_training_data,boston_training_data_target))
    np.random.shuffle(joint_data_array)
    boston_training_data_target = joint_data_array[:,13]
    boston_training_data = np.delete(joint_data_array, 13, 1)
    
    lamda = 0.0001
    while lamda<10 :
    
        cur_train_dataset = np.array_split(boston_training_data, 10)
        cur_train_dataset_target = np.array_split(boston_training_data_target, 10)
        train_data_MSE = 0
        test_data_MSE = 0    
        for i in range(0,10):
            testing_section = cur_train_dataset[i]
            testing_section_target_data_set = cur_train_dataset_target[i]
            training_section = np.concatenate(np.delete(cur_train_dataset, i, 0))
            training_target_section = np.concatenate(np.delete(cur_train_dataset_target, i, 0))
            data_set_dict = dict()
            data_set_dict[train_data] = training_section
            data_set_dict[test_data] = testing_section
            
            final_theta = ridge_estimator(data_set_dict[train_data], training_target_section, lamda)
            residue_array = predict_value(final_theta, data_set_dict[train_data], training_target_section)
            train_data_MSE = train_data_MSE + calculate_MSE(residue_array)
            testing_residue_array = predict_value(final_theta, data_set_dict[test_data], testing_section_target_data_set)
            test_data_MSE = test_data_MSE + calculate_MSE(testing_residue_array)
        MSE = test_data_MSE/10
        if MSE<old_MSE:
            old_MSE = MSE
            old_lamda = lamda
        lamda = lamda + 0.01
    print '\n', 'winning lamda:::::', old_lamda
    
    final_theta = ridge_estimator(boston_training_data, boston_training_data_target, old_lamda)
    residue_array = predict_value(final_theta, boston_training_data, boston_training_data_target)
    print 'MSE for CV Training Data :::',calculate_MSE(residue_array)
        
    residue_array = predict_value(final_theta, boston_testing_data, boston_testing_data_target)
    print 'MSE for CV Testing Data :::',calculate_MSE(residue_array)
    print '\n'
    
