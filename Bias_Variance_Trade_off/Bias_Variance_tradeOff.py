import numpy as np
import LinearRegression
import matplotlib.pyplot as plt

num_of_datasets = 100
num_of_points_in_dataset = 10
num_of_test_points = 1000
train_dataset_input_x = []
train_dataset_output_y = []
test_dataset_input_x = []
test_dataset_output_y = []


"""

  These methods: generate_train_dataset, generate_test_dataset, generate_data_set, generate_uniform_rand_num, generate_normal_rand_num
  
  Help in generating Training Data set, Testing Data set based on the values defined in gloabl variables : num_of_datasets, num_of_points_in_dataset
  The data generate is a uniform distribution data b/w -1 to 1, with a gaussain error added to it.
  True function : f(x) =y = 2(x^2) + epsilon(gaussian error), x is generated based on U.D. b/w (-1,1)
  
"""
def generate_train_dataset():
    
    global train_dataset_input_x, train_dataset_output_y
    train_dataset_input_x = np.empty([num_of_datasets, num_of_points_in_dataset])
    train_dataset_output_y = np.empty([num_of_datasets, num_of_points_in_dataset])
    data_set_dict = generate_data_set(train_dataset_input_x, train_dataset_output_y)
    train_dataset_input_x = data_set_dict['dataset_input_x']
    train_dataset_output_y = data_set_dict['dataset_output_y']
    
        
def generate_test_dataset():
    
    global test_dataset_input_x, test_dataset_output_y
    test_dataset_input_x = np.empty([1, num_of_test_points])
    test_dataset_output_y = np.empty([1, num_of_test_points])
    #isPrint = True
    data_set_dict = generate_test_data_set(test_dataset_input_x, test_dataset_output_y)
    test_dataset_input_x = data_set_dict['dataset_input_x']
    test_dataset_output_y = data_set_dict['dataset_output_y']

def generate_data_set(dataset_input_x, dataset_output_y):
    
    data_set_dict = {}
    for num_dataSet in range(0,num_of_datasets):
        ## Generate Input variable X
        dataset_input_x[num_dataSet] = generate_uniform_rand_num(-1,1)
        ## Generate output variable Y
        for datapoint in range(0,num_of_points_in_dataset):
            gaussain_error = generate_normal_rand_num(0,0.1,1)
            dataset_output_y[num_dataSet][datapoint] = 2*((dataset_input_x[num_dataSet][datapoint])**2) + gaussain_error
    data_set_dict['dataset_input_x'] = dataset_input_x
    data_set_dict['dataset_output_y'] = dataset_output_y
    return data_set_dict

def generate_test_data_set(dataset_input_x, dataset_output_y):
    
    data_set_dict = {}
    for num_dataSet in range(0,1):
        ## Generate Input variable X
        dataset_input_x[num_dataSet] = np.random.uniform(-1,1,num_of_test_points)
        ## Generate output variable Y
        for datapoint in range(0,num_of_test_points):
            gaussain_error = generate_normal_rand_num(0,0.1,1)
            dataset_output_y[num_dataSet][datapoint] = 2*((dataset_input_x[num_dataSet][datapoint])**2) + gaussain_error
    data_set_dict['dataset_input_x'] = dataset_input_x
    data_set_dict['dataset_output_y'] = dataset_output_y
    return data_set_dict   
                       
def generate_uniform_rand_num(start,end):
    return np.random.uniform(start,end,num_of_points_in_dataset)    

def generate_normal_rand_num(mu,sigma,num):    
    return np.random.normal(mu, sigma, num) 
 
"""
    Data set generation Methods END here.
""" 
 
 
 
 
"""
    feature_arr: feature_arr is the Traing Data X, in the form of dictionary where values are dataset wise feature array, expecting vectors of training examples in N*M form,
    where N = Number of training example, M = Number of features
    If the model is of the form, g(x) = wo, with no dependency on x,
    The method expect an empty array []
    
    regression_Lamda, is the lamda which is used for regularization when the model is trained using Linear Regression
    
    Returns: A list of Models in Models_dict['Models'], trained for the given input data array. 
    Number of model arrays returned depends on the number of datasets defined in the global variable num_of_datasets
    
"""
def train_dataset_based_models(feature_arr, regression_Lamda):     
    global train_dataset_input_x, train_dataset_output_y

    models_dict = {}
    theta = []
    MSE_arr = []

    for num_of_data_set in range(0,num_of_datasets):        
        if(None != feature_arr):
            cur_theta = LinearRegression.estimate_LR_theta_2(feature_arr[num_of_data_set], train_dataset_output_y[num_of_data_set], regression_Lamda)
            theta.append(cur_theta)
            model_base_predicted_value = LinearRegression.predict_value(cur_theta, feature_arr[num_of_data_set], train_dataset_output_y[num_of_data_set])
        else:
            theta = None
            model_base_predicted_value = np.ones((np.shape(train_dataset_output_y)[1],), dtype=np.int)
            
        MSE = LinearRegression.calculate_mean_square_error(model_base_predicted_value, train_dataset_output_y[num_of_data_set])
        MSE_arr.append(MSE)
    models_dict['Models'] = theta
    models_dict['MSE_arr'] = MSE_arr
    return models_dict

def calculate_bias_varaince(models, test_feature_arr):
    
    bias_var_dict = {}
    #print ('Calculating bias & variance for the test data set.')
    
    """
        model_base_predicted_value is an array of size N*M, where N is the number of testing examples, and M is the number of Models
        So, this array has the predicted values of each data point based on different models
    """
    if None != test_feature_arr:
        model_base_predicted_value = LinearRegression.predict_value(models, test_feature_arr['train_feature'], test_dataset_output_y[0])
        best_modelavg_estimated_value = np.mean(model_base_predicted_value, 1)
    else:
        model_base_predicted_value = np.ones((np.shape(test_dataset_output_y)[1],), dtype=np.int)
        best_modelavg_estimated_value = model_base_predicted_value
    
    bias_var_dict['BIAS'] = calculate_bias(best_modelavg_estimated_value)
    #bias_var_dict['BIAS'] = pow(calculate_bias(best_modelavg_estimated_value),2)
    bias_var_dict['VARIANCE'] = calculate_variance(model_base_predicted_value, best_modelavg_estimated_value)
    
    return bias_var_dict


def  calculate_bias(best_modelavg_estimated_value):
    bias = (best_modelavg_estimated_value - test_dataset_output_y[0]) ** 2
    return np.mean(bias)


def calculate_variance(model_base_predicted_value, best_modelavg_estimated_value): 
    pred_value = np.transpose(model_base_predicted_value)
    pred_value_diff = np.subtract(pred_value, best_modelavg_estimated_value) ** 2
    return np.mean(np.mean(pred_value_diff))

 
def generate_feature_vector(train_input_var, num_of_features):
        
    if(num_of_features == 0):
        return None
    elif(num_of_features == 1):
        feature_dict = {}
        feature_dict['test_feature'] = []
        feature_dict['train_feature'] = []
        return feature_dict
    else:
        train_feature_arr = []
        for i in range(1,num_of_features):
            if i == 1:
                train_feature_arr = np.power(train_input_var, i)
            else:
                train_feature_arr = np.column_stack((train_feature_arr,np.power(train_input_var, i)))   
        
        feature_dict = {}
        feature_dict['train_feature'] = train_feature_arr
        return feature_dict

def data_analysis_bias_variance(num_of_features_arr, lamda, dataset_count, datapoints_count, isPlot ):
    
    global num_of_dataset, num_of_points_in_dataset 
    num_of_dataset = dataset_count
    num_of_points_in_dataset = datapoints_count
    
    generate_train_dataset()
    generate_test_dataset()
    MSE_Dict = {}
    for lamda in lamda:
        print '\n\nLAMDA VALUE::::', lamda
        #Step 1: for each function do something
        i = 0
        for num_of_features in num_of_features_arr:
            dataset_wise_feature_dict = {}
            for num_of_data_set in range(0,num_of_datasets):
                transformed_feature_dict = generate_feature_vector(train_dataset_input_x[num_of_data_set], num_of_features)
                
                if(None != transformed_feature_dict):
                    if(len(transformed_feature_dict['train_feature']) > 0):
                        feature_arr = transformed_feature_dict['train_feature']
                        dataset_wise_feature_dict[num_of_data_set] = feature_arr
                    else:  dataset_wise_feature_dict[num_of_data_set] = []
                else:
                    dataset_wise_feature_dict = None
                    break
            
            print 'Number of parameters in function', num_of_features 
            models_dict = train_dataset_based_models(dataset_wise_feature_dict, lamda)
            models = models_dict['Models']
            MSE_Dict[num_of_features] = models_dict['MSE_arr']        
            #print 'Model Array shape', np.shape(models)
            test_data = generate_feature_vector(test_dataset_input_x[0], num_of_features)
            #print 'np.shape(test_data) of transformed test data', np.shape(test_data)
            bias_var_dict = calculate_bias_varaince(models, test_data)
            print 'Bias', bias_var_dict['BIAS']
            print 'Variance', bias_var_dict['VARIANCE'], '\n\n'
            
        if isPlot:
            plotFeatureSubplots(MSE_Dict)

def plotFeatureSubplots(MSE_Dict):
    
    global correlation_vector
    plt.figure(1)
    plt.subplots_adjust(hspace=1)
    
    key_list = MSE_Dict.keys()
    i = 0
    for key in key_list:
        i += 1
        feature_subplot = plt.subplot(4, 4, i)
        label = 'g'+str(key+1)+'(x)'
        feature_subplot.set_title(label) 
        plt.hist(MSE_Dict[key], bins=20)
    plt.show()
    
#test_scripts()
#generate_train_dataset()