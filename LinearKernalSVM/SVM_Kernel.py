import scipy.io as sio
import numpy as np
from svmutil import *
import timeit

train_data_set = []
test_data_set = []

def load_dataset():
     
    global train_data_set, test_data_set
    train_data_set = sio.loadmat('phishing-train.mat')
    test_data_set = sio.loadmat('phishing-test.mat')
    
    
def train_svm_model():
    global train_data_set, test_data_set
    y, x = np.transpose(train_data_set['label']), train_data_set['features'].tolist()
    y_test, x_test = np.transpose(train_data_set['label']), train_data_set['features'].tolist()
    time_arr = []
    accuracy_arr = []
    for pwr in range(-6,3):
        
        C = pow(4,pwr)
        prob = svm_problem(y, x)
        param = svm_parameter('-s 0 -c '+str(C)+' -v 3 -q')
        start_time = timeit.default_timer()
        m = svm_train(prob, param)
        accuracy_arr.append(m)
        elapsed = timeit.default_timer() - start_time
        time_arr.append(elapsed / 3)
    
    for i in range(0,len(time_arr)):   
        print pow(4,(-6+i)), ' :: ',accuracy_arr[i],' :: ', time_arr[i]
        #print 'accuracy_arr :::', 
    
def train_kernelised_svm_model():
    global train_data_set, test_data_set
    y, x = np.transpose(train_data_set['label']), train_data_set['features'].tolist()
    y_test, x_test = np.transpose(train_data_set['label']), train_data_set['features'].tolist()
    time_arr_poly = []
    time_arr_rbf = []
    accuracy_arr_poly = []
    C_arr_poly = []
    degree_arr_poly = []
    accuracy_arr_RBF = []
    C_arr_rbf = []
    gamma_arr_rbf= []
    gama_p_arr = []
    for kernel_type in [1, 2]:
        if kernel_type == 1:
                print 'Polynomial'
        else: print 'RBF'
        for pwr in range(-3,8):
            
            C = pow(4,pwr)
            #print '\n\nC VALUE:', C
            prob = svm_problem(y, x)
            param_str_comon = '-c '+str(C)+' -v 3 -q'
            if kernel_type == 1:
                
                for degree in [1,2,3]:
                    param_str = param_str_comon +' -t 1 -d '+ str(degree) 
                    #print param_str
                    start_time = timeit.default_timer()
                    accuracy = svm_train(prob, param_str)
                    accuracy_arr_poly.append(accuracy)
                    C_arr_poly.append(C)
                    degree_arr_poly.append(degree)
                    elapsed = timeit.default_timer() - start_time
                    #print(end - start)
                    time_arr_poly.append(elapsed / 3)
            if kernel_type == 2:
                
                for gamma_p in range(-7,0):
                    gamma = pow(4,gamma_p)
                    param_str = param_str_comon +' -t 2 -g '+ str(gamma) 
                    start_time = timeit.default_timer()
                    accuracy_RBF = svm_train(prob, param_str)
                    accuracy_arr_RBF.append(accuracy_RBF)
                    C_arr_rbf.append(C)
                    gama_p_arr.append(gamma)
                    elapsed = timeit.default_timer() - start_time
                    #print(end - start)
                    time_arr_rbf.append(elapsed / 3)
    
    for i in range(0,len(C_arr_poly)):
        print C_arr_poly[i], '::', degree_arr_poly[i], '::',  time_arr_poly[i], '::',  accuracy_arr_poly[i]
    
    print '\n\n'   
    for i in range(0,len(C_arr_rbf)):
        print C_arr_rbf[i], '::', gama_p_arr[i], '::',  time_arr_rbf[i], '::',  accuracy_arr_RBF[i]

    
    
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




#plotFeatureSubplots()