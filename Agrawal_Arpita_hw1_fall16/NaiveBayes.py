import numpy as np
import collections
from scipy import stats
ST_DV = "Standard_Deviation"
Mean = "Mean"

def NB():
    
    print ('\n\nEstimating class based on Naive Bayes Algorithm :: \n\n')
    
    trainDataSet = np.genfromtxt("train.txt",delimiter=",",usecols=range(1 , 11))
    testingDataSet = np.genfromtxt("test.txt",delimiter=",",usecols=range(1 , 11))
    trainDataSet[trainDataSet[:, -1].argsort()]
      
    estimatedGaussianParams = {}
    estimatedGaussianParams[Mean] = np.array([a.mean(0) for a in np.split(trainDataSet, np.argwhere(np.diff(trainDataSet[:, -1])) + 1)])   
    estimatedGaussianParams[ST_DV] = np.column_stack((np.array([data.std(0, ddof = 1) for data in np.split(trainDataSet, np.argwhere(np.diff(trainDataSet[:,-1])) + 1)]), np.unique(trainDataSet[:, -1])))
    
    print ('\nTesting Data Accuracy on Naive Bayes Algorithm :: ')
    printAccuracy(estimatedGaussianParams, testingDataSet, trainDataSet)
    print ('\nTraining Data Accuracy on Naive Bayes Algorithm :: ')
    printAccuracy(estimatedGaussianParams, trainDataSet, trainDataSet)

def printAccuracy(estimatedGaussianParams, dataset, trainDataSet):
    count = 0
    class_dict = probability_each_class(trainDataSet)
    for data in dataset:
        isCorrect = predictValueBasedOnGaussianModel(estimatedGaussianParams, data, trainDataSet, class_dict)
        if isCorrect:
            count = count +1
    print float(count)/float(np.shape(dataset)[0])
    
def predictValueBasedOnGaussianModel(estimatedGaussianParams, testDataSet, trainDataSet, class_dict):
    
    prob = 1
    maxProb = 0
    maxProbClass = ''
    for x in range(0, np.shape(estimatedGaussianParams[ST_DV])[0]):
        prob = 1
        for y in range(0,len(testDataSet) -1):
            mean = estimatedGaussianParams[Mean][x][y]
            std_dev = estimatedGaussianParams[ST_DV][x][y]
            if mean==0:
                if testDataSet[y]==0:   prob *= 1
                else: prob *= 0
            else:   prob =prob*stats.norm.pdf(testDataSet[y],mean, std_dev)        
        prob = prob*class_dict[estimatedGaussianParams[Mean][x][-1]]
        if prob> maxProb:
            maxProb = prob
            maxProbClass = estimatedGaussianParams[Mean][x][-1] 
    return maxProbClass == testDataSet[-1]

    
def probability_each_class(trainDataSet):  
    tot_elements = trainDataSet.shape[0]
    classes_dict = collections.Counter(trainDataSet[:,-1])
    for key in classes_dict.keys():
        classes_dict[key] = float(classes_dict[key])/float(tot_elements)
    return classes_dict
