import numpy as np
import LinearRegression
import matplotlib.pyplot as plt

num_of_datasets = 100
num_of_points_in_dataset = 10
train_dataset_input_x = []
train_dataset_output_y = []

def generate_train_dataset():
    
    global train_dataset_input_x, train_dataset_output_y
    train_dataset_input_x = np.empty([num_of_datasets, num_of_points_in_dataset])
    train_dataset_output_y = np.empty([num_of_datasets, num_of_points_in_dataset])
    #isPrint = True
    for num_dataSet in range(0,num_of_datasets):
        ## Generate Input variable X
        train_dataset_input_x[num_dataSet] = generate_uniform_rand_num(-1,1)
        ## Generate output variable Y
        for datapoint in range(0,num_of_points_in_dataset):
            gaussain_error = generate_normal_rand_num(0,0.1,1)
            train_dataset_output_y[num_dataSet][datapoint] = 2*((train_dataset_input_x[num_dataSet][datapoint])**2) + gaussain_error
        
    
def generate_uniform_rand_num(start,end):
    return np.random.uniform(start,end,num_of_points_in_dataset)    

def generate_normal_rand_num(mu,sigma,num):    
    return np.random.normal(mu, sigma, num) 
 
def generate_feature_vector(train_input_var, num_of_features):
        
    if(num_of_features == 0):
        return None
    elif(num_of_features == 1):
        return []
    else:
        feature_arr = []
        for i in range(1,num_of_features):
            if i == 1:
                feature_arr = np.power(train_input_var, i)
            else:
                feature_arr = np.column_stack((feature_arr,np.power(train_input_var, i)))   
        return feature_arr

def data_analysis(lamda, num_of_features_arr, isPlot):
    
    global train_dataset_input_x, train_dataset_output_y
   # print "Data analysis based on different functions "
    
    MSE_Dict = dict()
    loss_arr_Dict = dict()
    estimated_value_arr_dict = dict()
    lamda_data_dict = {}
    for lamda in lamda:
        print '\n\nLAMDA VALUE::::', lamda
        #Step 1: for each function do something
        i = 0
        for num_of_features in num_of_features_arr:
            MSE_arr = []
            loss_arr = np.empty([num_of_datasets, num_of_points_in_dataset])
            estimated_value_arr = []
            #Step 2: For each dataset do something
            for num_of_data_set in range(0,num_of_datasets):
                # Step 3: generate feature vector
                feature_arr = generate_feature_vector(train_dataset_input_x[num_of_data_set], num_of_features)
                if(None != feature_arr):
                    # Step 4: estimate theta
                    theta = LinearRegression.estimate_LR_theta_2(feature_arr, train_dataset_output_y[num_of_data_set], lamda)
                    # Step 5: predict y
                    estimated_value = LinearRegression.predict_value(theta, feature_arr, train_dataset_output_y[num_of_data_set])
                else:
                    estimated_value = np.ones((np.shape(train_dataset_output_y)[1],), dtype=np.int)
                
                if np.size(estimated_value_arr) <1:
                    estimated_value_arr = estimated_value
                else: estimated_value_arr = np.append(estimated_value_arr,estimated_value)
                
                loss_arr[num_of_data_set] = LinearRegression.calculate_mean_square_error(estimated_value, train_dataset_output_y[num_of_data_set])
                # Step 6: Calculate mean square error
                MSE = LinearRegression.calculate_mean_square_error(estimated_value, train_dataset_output_y[num_of_data_set])
                MSE_arr.append(MSE)
            MSE_Dict[num_of_features] = MSE_arr
            loss_arr_Dict[num_of_features] = loss_arr
            estimated_value_arr_dict[num_of_features] = estimated_value_arr  
            #print 'shape_loss_dict', np.shape(loss_arr_Dict[num_of_features])
        #Step 7: Plot histogram of 100 MSE for each function
        if isPlot:
            plotFeatureSubplots(MSE_Dict)
        
        bias = calculate_bias(loss_arr_Dict)
        print 'bias::', bias
        varaince = calculate_variance(estimated_value_arr_dict)
        print 'variance::', varaince
        bias_var = {}
        bias_var['bias'] = bias
        bias_var['variance'] = varaince
        lamda_data_dict[lamda] = bias_var
        i += 1
    return lamda_data_dict

def calculate_bias(loss_arr_Dict):
    keys = loss_arr_Dict.keys()
    bias = []
    for key in keys:
        loss_arr_dataset = loss_arr_Dict[key]
        mean_arr = np.mean(loss_arr_dataset, 1)
        mean_s = np.mean(mean_arr, 0)
        bias.append(mean_s)
    return bias

#TODO::: Mean calculation has to be done- dataset wise, or mean of 1000 data point is cool>???
def calculate_variance(estimated_value_arr_dict):
    keys = estimated_value_arr_dict.keys()
    variance_arr = []
    variance = []
    i = 0
    for key in keys: 
        i += 1
        mean_dataset = np.mean(estimated_value_arr_dict[key]) 
        variance = (estimated_value_arr_dict[key]- mean_dataset) ** 2
        variance_arr.append(np.mean(variance))
    return variance_arr

def plotFeatureSubplots(MSE_Dict):
    
    global correlation_vector
    plt.figure(1)
    plt.subplots_adjust(hspace=1)
    
    key_list = MSE_Dict.keys()
    i = 0
    for key in key_list:
        i += 1
        feature_subplot = plt.subplot(4, 4, i)
        feature_subplot.set_title(key) 
        plt.hist(MSE_Dict[key], bins=10)
    plt.show()
        
####Test script
def test_scripts():
    #s = np.random.uniform(-1,1,num_of_points_in_dataset)
    print generate_normal_rand_num(0, 0.1, 1)

    for i in range(0,10):
        arr = generate_uniform_rand_num(-1,1)
        #print arr
        print np.mean(arr), "\n"
    
#test_scripts()
#generate_train_dataset()







def train_dataset_based_models(feature_arr, regression_Lamda):
    print 'Dataset based models'
    global train_dataset_input_x, train_dataset_output_y
#     estimated_value_arr = []
#     MSE_arr = []
#     loss_arr = np.empty([num_of_datasets, num_of_points_in_dataset])
    Models_dict = {}
    theta = []
    for num_of_data_set in range(0,num_of_datasets):
        
        if(None != feature_arr):
            # Step 4: estimate theta
            theta = LinearRegression.estimate_LR_theta_2(feature_arr, train_dataset_output_y[num_of_data_set], regression_Lamda)
            # Step 5: predict y
            #estimated_value = LinearRegression.predict_value(theta, feature_arr, train_dataset_output_y[num_of_data_set])
        else:
            theta = None
            #estimated_value = np.ones((np.shape(train_dataset_output_y)[1],), dtype=np.int)
#         if np.size(estimated_value_arr) <1:
#             estimated_value_arr = estimated_value
#         else: estimated_value_arr = np.append(estimated_value_arr,estimated_value)
#                 
#         loss_arr[num_of_data_set] = LinearRegression.calculate_mean_square_error(estimated_value, train_dataset_output_y[num_of_data_set])
#         # Step 6: Calculate mean square error
#         MSE = LinearRegression.calculate_mean_square_error(estimated_value, train_dataset_output_y[num_of_data_set])
#         MSE_arr.append(MSE)
    #Models_dict['MSE_arr'] = MSE_arr
    #Models_dict['loss_arr'] = loss_arr
    #Models_dict['estimated_value_arr_dict'] = estimated_value_arr
    Models_dict['theta'] = theta
    #MSE_Dict[num_of_features] = MSE_arr
    #loss_arr_Dict[num_of_features] = loss_arr
    #estimated_value_arr_dict[num_of_features] = estimated_value_arr 
    
    
    
    
def data_analysis(lamda, num_of_features_arr, isPlot):
    
    global train_dataset_input_x, train_dataset_output_y
    # print "Data analysis based on different functions "
    
    MSE_Dict = dict()
    loss_arr_Dict = dict()
    estimated_value_arr_dict = dict()
    lamda_data_dict = {}
    for lamda in lamda:
        print '\n\nLAMDA VALUE::::', lamda
        #Step 1: for each function do something
        i = 0
        for num_of_features in num_of_features_arr:
            MSE_arr = []
            loss_arr = np.empty([num_of_datasets, num_of_points_in_dataset])
            estimated_value_arr = []
            #Step 2: For each dataset do something
            for num_of_data_set in range(0,num_of_datasets):
                # Step 3: generate feature vector
                feature_arr = generate_feature_vector(train_dataset_input_x[num_of_data_set], num_of_features)
                if(None != feature_arr):
                    # Step 4: estimate theta
                    theta = LinearRegression.estimate_LR_theta_2(feature_arr, train_dataset_output_y[num_of_data_set], lamda)
                    # Step 5: predict y
                    estimated_value = LinearRegression.predict_value(theta, feature_arr, train_dataset_output_y[num_of_data_set])
                else:
                    estimated_value = np.ones((np.shape(train_dataset_output_y)[1],), dtype=np.int)
                
                if np.size(estimated_value_arr) <1:
                    estimated_value_arr = estimated_value
                else: estimated_value_arr = np.append(estimated_value_arr,estimated_value)
                
                loss_arr[num_of_data_set] = LinearRegression.calculate_mean_square_error(estimated_value, train_dataset_output_y[num_of_data_set])
                # Step 6: Calculate mean square error
                MSE = LinearRegression.calculate_mean_square_error(estimated_value, train_dataset_output_y[num_of_data_set])
                MSE_arr.append(MSE)
            MSE_Dict[num_of_features] = MSE_arr
            loss_arr_Dict[num_of_features] = loss_arr
            estimated_value_arr_dict[num_of_features] = estimated_value_arr  
            #print 'shape_loss_dict', np.shape(loss_arr_Dict[num_of_features])
        #Step 7: Plot histogram of 100 MSE for each function
        if isPlot:
            plotFeatureSubplots(MSE_Dict)
        
        bias = calculate_bias(loss_arr_Dict)
        print 'bias::', bias
        varaince = calculate_variance(estimated_value_arr_dict)
        print 'variance::', varaince
        bias_var = {}
        bias_var['bias'] = bias
        bias_var['variance'] = varaince
        lamda_data_dict[lamda] = bias_var
        i += 1
    return lamda_data_dict




def generate_feature_vector(train_input_var,test_input_var, num_of_features):
        
    if(num_of_features == 0):
        return None
    elif(num_of_features == 1):
        feature_dict = {}
        feature_dict['test_feature'] = []
        feature_dict['train_feature'] = []
        return feature_dict
    else:
        train_feature_arr = []
        test_feature_arr = []
        for i in range(1,num_of_features):
            if i == 1:
                test_feature_arr = np.power(test_input_var, i)
                train_feature_arr = np.power(train_input_var, i)
            else:
                test_feature_arr = np.column_stack((test_feature_arr,np.power(test_input_var, i)))
                train_feature_arr = np.column_stack((train_feature_arr,np.power(train_input_var, i)))   
        
        feature_dict = {}
        feature_dict['test_feature'] = test_feature_arr
        feature_dict['train_feature'] = train_feature_arr
        return feature_dic

     
