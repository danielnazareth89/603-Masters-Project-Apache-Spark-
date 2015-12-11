import sys
import time
from random import random
from pyspark import SparkContext
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from math import log
import time
from math import exp
from collections import defaultdict
import hashlib

def loadData():
    
    sc = SparkContext(appName="SparkPi")
    
    rawData=(sc.textFile('/gpfs/courses/cse603/students/CTR/train_80M')
    #rawData=(sc.textFile('/user/meethilv/PDP/sample_train')
    #rawData=(sc.textFile('/scratch/CTR/train1')
             #.map(lambda x: x.replace(',','==='))
             )
    return rawData

def hashFunction(numBuckets, rawFeats, printMapping=False):
    mapping = {}
    for ind, category in rawFeats:
        featureString = category + str(ind)
        mapping[featureString] = int(int(hashlib.md5(featureString).hexdigest(), 16) % numBuckets)
    if(printMapping): print mapping
    sparseFeatures = defaultdict(float)
    for bucket in mapping.values():
        sparseFeatures[bucket] += 1.0
    return dict(sparseFeatures)

def parsePoint(row):
    
    features = row.split(',')[2:]
    indexed_features=[]
    for i in range(0,len(features)):
        indexed_features.append((i,features[i]))
        
    return indexed_features

def parseHashPoint(point, numBuckets):

    features=parsePoint(point)
    print "len(features)"+str(len(features))
    points_list=point.split(',')
    label=points_list[1]
    
    hashedFeatures=hashFunction(numBuckets,features,printMapping=False)
    
    return LabeledPoint(label,SparseVector(numBuckets,hashedFeatures))


def computeLogLoss(p, y):
    epsilon = 10e-12
    if p==0:
        p=p+epsilon
    elif p==1:
        p=p-epsilon
    if y==1:
        return -log(p)
    elif y==0:
        return -log(1-p)



def splitData(rawData):
    
    weights = [0.8,0.2]
    seed = 234
    
    rawTrainData,rawValidationData = rawData.randomSplit(weights,seed)
    
    #rawTrainData.cache()                            ############    check if without
    #rawValidationData.cache()
    
    #print rawTrainData.count()
    #print rawValidationData.count()
    
    return rawTrainData,rawValidationData

    
def getLogisticRegressionModel(Train_Data):  
    
    numIters = 10
    stepSize = 10.
    regParam = 1e-6
    regType = 'l2'
    includeIntercept = True
    
    
    return LogisticRegressionWithSGD.train(data = Train_Data,
                                   iterations = numIters,
                                   miniBatchFraction=0.1,
                                   step = stepSize,
                                   regParam = regParam,
                                   regType = regType,
                                   intercept = includeIntercept)
    
    
def getPrediction(x, w, intercept):

    rawPrediction = x.dot(w) + intercept

    # Bound the raw prediction value
    rawPrediction = min(rawPrediction, 20)
    
    rawPrediction = max(rawPrediction, -20)
    return 1.0 / (1.0 + exp(-rawPrediction))

def evaluateResults(model, data):
    return data.map(lambda x: computeLogLoss(getPrediction(x.features, model.weights, model.intercept), x.label)).sum() / data.count()


if __name__ == "__main__":
    
    start = time.time()
    raw_Data = loadData()
    end = time.time()
    print "Time for loadData = "+str(end-start)
    
    start = time.time()
    raw_Train_Data,raw_Validation_Data = splitData(raw_Data)
    end = time.time()
    print "Time for splitData = "+str(end-start)

    print "##############          feature hashing        ##############"
    
    start = time.time()
    numBucketsCTR = 2 ** 15
    hashTrainData = raw_Train_Data.map(lambda point: parseHashPoint(point,numBucketsCTR))
    hashTrainData.cache()
    end = time.time()
    print "Time for feature hashing featureExtraction = "+str(end-start)
    
    start = time.time()
    TrainDataLRModel = getLogisticRegressionModel(hashTrainData)
    end = time.time()
    print "Time for getLogisticRegressionModel = "+str(end-start)
    
    
    start = time.time()
    trainingPredictions = hashTrainData.map(lambda x: getPrediction(x.features, TrainDataLRModel.weights, TrainDataLRModel.intercept))
    end = time.time()
    print trainingPredictions.take(5)
    print "Time for trainingPredictions = "+str(end-start)
    
    
    classOneFracTrain = hashTrainData.map(lambda x: x.label).reduce(lambda a, b: a + b) / hashTrainData.count()
    print classOneFracTrain

    logLossTrBase = hashTrainData.map(lambda x: computeLogLoss(classOneFracTrain, x.label)).sum() / hashTrainData.count()
    print 'Baseline Train Logloss = {0:.3f}\n'.format(logLossTrBase)
    
    start = time.time()
    logLossTrLR0 = evaluateResults(TrainDataLRModel, hashTrainData)
    print ('Feature hashing Features Train Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'.format(logLossTrBase, logLossTrLR0))
    end = time.time()
    print "Time for evaluateResults = "+str(end-start)
    
    
