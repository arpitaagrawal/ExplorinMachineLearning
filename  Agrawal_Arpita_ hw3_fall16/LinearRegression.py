import numpy as np


def predict_value(theta2, dataArray, targetDataArray):
    num_test_examples = np.shape(targetDataArray)[0]
    column_of_ones = np.ones((num_test_examples), dtype=np.int)
    if np.shape(dataArray)[0]>0:
        joint_data_array =  np.column_stack((column_of_ones,dataArray))
    else:  
        joint_data_array =  column_of_ones
        joint_data_array = np.reshape(joint_data_array, (num_test_examples, 1))

    estimated_value = np.dot(joint_data_array, np.transpose(theta2))
    return estimated_value
    


def sum_square_arr(dataArray, targetDataArray):
    sum_sqr = np.sum(np.subtract(dataArray, targetDataArray) ** 2)
    return sum_sqr


def calculate_mean_square_error(estimated_value, targetDataArray):
    num_test_examples = np.shape(estimated_value)[0]
    
    loss = np.subtract(estimated_value, targetDataArray)
    train_data_MLE = float(1/(float(num_test_examples))) * np.sum(loss ** 2)
    return train_data_MLE
    
def create_lamda_matrix(size, lamda):
    lambda_vec = np.zeros((size, size))
    np.fill_diagonal(lambda_vec,lamda)
    lambda_vec[0][0] = 0
    return lambda_vec
    
def estimate_LR_theta_2(dataArray, targetDataArray, lamda):
    #print 'data array shape', np.shape(dataArray)
    num_test_examples = np.shape(targetDataArray)[0]
    
    column_of_ones = np.ones((num_test_examples,), dtype=np.int)
    
    if np.shape(dataArray)[0]>0:
        joint_data_array =  np.column_stack((column_of_ones,dataArray))
    else:  
        joint_data_array =  column_of_ones
        joint_data_array = np.reshape(joint_data_array, (num_test_examples, 1))
        
    num_of_features = np.shape(joint_data_array)[1]
    theta = np.dot(np.transpose(joint_data_array), joint_data_array)
    theta = theta + create_lamda_matrix(num_of_features, lamda)
    data_inv = np.linalg.pinv(np.matrix(theta))
    data_inv = np.dot(data_inv, np.transpose(joint_data_array))
    final_theta = np.dot(data_inv, targetDataArray)
    final_theta = np.array(final_theta).flatten()
    return final_theta

