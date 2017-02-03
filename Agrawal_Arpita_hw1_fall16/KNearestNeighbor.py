import numpy as np
EUCLEDIAN_DISTANCE = "Eucliedean_Distance"
MANHATTAN_DISTANCE = "Manhattan_Distance"

def KNN():
    
    print ('\n\nEstimating class based on K Nearest Neighbor Algorithm :: ')
    classesDictionary = {}
    
    trainDataSet = np.genfromtxt("train.txt",delimiter=",")
    testingDataSet = np.genfromtxt("test.txt",delimiter=",")
    
    normalizeParams(trainDataSet,testingDataSet)
    for ele in np.unique(trainDataSet[:,10]):
        classesDictionary[ele] = 0
    #classifyBasedOnKNearestNeghbor(3, EUCLEDIAN_DISTANCE, testingDataSet[0], trainDataSet, classesDictionary)
    for k in [1,3,5,7]:
        print '\n\nk = ', k
        print('L2 ( Eucledian ):')
        getAccuracy(k, testingDataSet, trainDataSet, classesDictionary, metric= EUCLEDIAN_DISTANCE)
        getTrainingAccuracy(k, trainDataSet, classesDictionary, metric= EUCLEDIAN_DISTANCE)
        
        print('L1( Manhattan )')
        getAccuracy(k, testingDataSet, trainDataSet, classesDictionary, metric= MANHATTAN_DISTANCE)
        getTrainingAccuracy(k, trainDataSet, classesDictionary, metric= MANHATTAN_DISTANCE)
        
        
def normalizeParams(trainDataSet,testingDataSet): 
    mean = trainDataSet.mean(0)
    
    std_1 = trainDataSet.std(0, ddof = 1)
    for i in range(0,np.shape(trainDataSet)[0]-1):
        for j in range(1,np.shape(trainDataSet)[1]-1):
            trainDataSet[i][j] = (trainDataSet[i][j] - mean[j])/std_1[j]
            
    for i in range(0,np.shape(testingDataSet)[0]-1):
        for j in range(1,np.shape(testingDataSet)[1]-1):
            testingDataSet[i][j] = (testingDataSet[i][j] - mean[j])/std_1[j]

def getAccuracy(k, testData, trainData, classesDictionary, metric):

    sizeOfTestData = len(testData)
    correctEstimations = 0
    incorrectEstimations = 0
    
    for i in range(0, sizeOfTestData):
        estimation = classifyBasedOnKNearestNeghbor(k, metric, testData[i], trainData, classesDictionary)
        #print 'estimation', estimation, testData[i][10]
        if(estimation == testData[i][10]):
            correctEstimations += 1
        else:
            incorrectEstimations += 1
    
    print 'Accuracy(Testing Data), ::', float(correctEstimations)/float(sizeOfTestData) 


def getTrainingAccuracy(k, trainData, classesDictionary, metric):
    
    correctEstimations = 0
    incorrectEstimations = 0
    
    for i in range(0,len(trainData)):
        dataCopy = np.delete(trainData, i, 0)
        estimation = classifyBasedOnKNearestNeghbor(k, metric, trainData[i], dataCopy, classesDictionary)
        #print 'estimation', estimation, testData[i][10]
        if(estimation == trainData[i][10]):
            correctEstimations += 1
        else:
            incorrectEstimations += 1
    print 'Accuracy(Training Data) ::', float(correctEstimations)/float(len(trainData))
            
        
def classifyBasedOnKNearestNeghbor(k, metric, datapoint, trainingSet, classesDictionary):
    #Step1 : Based on Metric find the distance of
    ##       the point from all the points in the training set
    sortedDistanceArr = []
    if metric == EUCLEDIAN_DISTANCE:
        sortedDistanceArr = findEuclideanDistance(datapoint, trainingSet)
    elif metric == MANHATTAN_DISTANCE:
        sortedDistanceArr = finnManhattanDistance(datapoint, trainingSet)
    estimation = classifyBasedOnMajority(k,sortedDistanceArr, classesDictionary.copy(), trainingSet)
    
    return estimation    
    #Step2 :     
    
def classifyBasedOnMajority(k,sortedDistanceArr, classesDictionary, trainingSet):
    for num in range(0,k):
        index = int(sortedDistanceArr[num][0])
        index2 = int(np.where(trainingSet[:, 0] == index)[0])
        classesDictionary[trainingSet[index2][10]] = int(classesDictionary[trainingSet[index2][10]]) + 1
    maxV = 0
    maxKey = ''  
    
    for key in classesDictionary:
        if maxV< int(classesDictionary[key]):
            maxV = int(classesDictionary[key])
            maxKey = key   
    return maxKey

def findEuclideanDistance(point, trainingSet):
    
    dimension = len(point)
    distanceArr = []
    for trainD in trainingSet:
        euclDistance = 0
        for i in range(1,dimension-1):
            euclDistance += pow(float(trainD[i]) - float(point[i]), 2)
        
        distanceArr.append([trainD[0],np.sqrt(euclDistance)])    
    return sorted(distanceArr, key=getValue)

def finnManhattanDistance(point, trainingSet):
    
    dimension = len(point)
    distanceArr = []
    for trainD in trainingSet:
        manhattanDistance = 0
        for i in range(1,dimension-1):
            manhattanDistance += abs(float(trainD[i]) - float(point[i]))
            
        distanceArr.append([trainD[0],manhattanDistance])    
    return sorted(distanceArr, key=getValue)    


def getValue(item):
    return item[1]
      