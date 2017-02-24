import scipy.io as sio
import numpy as np
import timeit
from svmutil import *


train_data_set = []
test_data_set = []

def load_dataset():
     
    global train_data_set, test_data_set
    train_data_set = sio.loadmat('phishing-train.mat')
    test_data_set = sio.loadmat('phishing-test.mat')
    
    
    
def data_processing():
    
    global train_data_set
    global test_data_set

    for i in range(0,np.shape(train_data_set['features'])[1]):
        
        unique_values_train = np.unique(train_data_set['features'][:,i])
        num_of_examples = len(train_data_set['features'][:,i])
        #print 'i:::', i, '::', unique_values_train 
        if len(unique_values_train) > 2:
            for val in unique_values_train:
                    #create new column
                new_test_col = []
                new_train_col = []
                #check if train and test is same val
                for num in range(0, num_of_examples):
                    if train_data_set['features'][num][i] == val: new_train_col.append(1)
                    else: new_train_col.append(0)
                    if test_data_set['features'][num][i] == val: new_test_col.append(1)
                    else: new_test_col.append(0)
                train_data_set['features'] = np.column_stack((train_data_set['features'],new_train_col))
                test_data_set['features'] = np.column_stack((test_data_set['features'],new_test_col))

        for num in range(0, num_of_examples):
            if train_data_set['features'][num][i] == -1: 
                train_data_set['features'][num][i] = 0
            if test_data_set['features'][num][i] == -1: 
                test_data_set['features'][num][i] = 0
    
    train_data_set['features'] = np.delete(train_data_set['features'],[1, 6, 7, 13, 14, 25, 28],1)
    test_data_set['features'] = np.delete(test_data_set['features'],[1, 6, 7, 13, 14, 25, 28],1)    


load_dataset()    
data_processing()
y, x = np.transpose(train_data_set['label']), train_data_set['features'].tolist()
y_test, x_test = np.transpose(test_data_set['label']), test_data_set['features'].tolist()
prob = svm_problem(y, x)
param_str = svm_parameter('-s 0 -t 2 -c {0} -g {1} -q'.format(1,4**-1))
#param_str = param_str_comon +' -t 2 -g 0.0625' 
start_time = timeit.default_timer()
model = svm_train(prob, param_str)
svm_predict(y_test, x_test, model)

elapsed = timeit.default_timer() - start_time