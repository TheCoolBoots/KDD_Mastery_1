from email.utils import collapse_rfc2231_value
import math
import pandas as pd
import numpy as np

# IS THIS FUNCTION CORRECT
def getGaussianLikelihood(x, mean, sd):
    # using this website formula: http://hyperphysics.phy-astr.gsu.edu/hbase/Math/gaufcn.html
    return (2 * math.pi * sd ** 2) ** -.5 * math.e ** (-.5 * ((x-mean)/sd) ** 2)  
    

def computePriors(y):
    valueCounts = y.value_counts().sort_index()
    priors = {f'{y.name}={item[0]}': item[1]/valueCounts.sum() for item in valueCounts.iteritems()}
    return priors


# create a dictionary that maps
# {feature name : {target value : (mean, sd)}}

# for each feature f:
#       for each target value v:
#           get all entries where target t = v
#           dict[f][v] = (mean, sd)
def getFeatureGaussianModels(X:pd.DataFrame, y:pd.Series):

    featureModels = {}
    for colName, col in X.iteritems():
        targetValues = {}
        for val in y.unique():
            targetValues[val] = (col[y==val].mean(), col[y==val].std())
        featureModels[colName] = targetValues

    return featureModels


# x is a series mapping colNames to values
def posteriors(gaussianModels:dict, priors:dict, x:pd.Series):

    numerators = {}

    """
    for every value v, i in priors:
        numerator[i] = priors[v]
        for every feature f, val in x:
            multiply numerator[i] by getGaussianLiklihood(gaussianModels[f][v], val)
        
    denominator = sum(numerators)
    """

    for priorVal, priorProb in priors.items():
        numerator = priorProb
        for colName, val in x.iteritems():
            numerator *= getGaussianLikelihood(val, *gaussianModels[colName][priorVal])
        numerators[priorVal] = numerator
    
    denominator = sum(numerators.values())
    post_priors = {}
    for priorVal, priorProb in priors.items():
        indexStr = None
        for colName, val in x.iteritems():
            if indexStr is None:
                indexStr = f'{priorVal}|{colName}={val}'
            else:
                indexStr += f',{colName}={val}'
        post_priors[indexStr] = numerators[priorVal]/denominator

    return post_priors
    

def train_test_split(X,y,test_frac=0.5):
    
    inxs = list(range(len(y)))
    np.random.shuffle(inxs)
    X = X.iloc[inxs,:]
    y = y.iloc[inxs]
    
    divPt = int(len(X)*test_frac) + 1
    Xtrain = X.iloc[:divPt]
    Xtest = X.iloc[divPt:]
    ytrain = y.iloc[:divPt]
    ytest = y.iloc[divPt:]

    return Xtrain,ytrain,Xtest,ytest