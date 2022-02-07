from audioop import avg
from email.utils import collapse_rfc2231_value
import math
from random import gauss
import pandas as pd
import numpy as np
from scipy.stats import norm



# IS THIS FUNCTION CORRECT
def getGaussianProbability(x, mean, sd):
    # norm.cdf is probability less than or equal to parameter x
    # uses a fraction of sd to get a range of x values
    b = x + sd*.2
    a = x - sd*.2
    return norm.cdf(b, loc=mean, scale=sd) - norm.cdf(a, loc=mean, scale=sd)
    

def computePriors(y):
    valueCounts = y.value_counts().sort_index()
    priors = {item[0]: item[1]/valueCounts.sum() for item in valueCounts.iteritems()}
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
def predictX(gaussianModels:dict, priors:dict, x:pd.Series):

    numerators = {}

    """
    for every value i, v in priors:
        numerator[i] = priors[v]
        for every feature f, val in x:
            multiply numerator[i] by getGaussianLiklihood(gaussianModels[f][v], val)
        
    denominator = sum(numerators)
    """

    for priorVal, priorProb in priors.items():
        numerator = priorProb
        for colName, val in x.iteritems():
            numerator *= getGaussianProbability(val, *(gaussianModels[colName][priorVal]))
        numerators[priorVal] = numerator
    
    denominator = sum(numerators.values())
    post_priors = {}
    max = (0, 0)
    for priorVal, priorProb in priors.items():
        post_priors[priorVal] = numerators[priorVal]/denominator
        if post_priors[priorVal] >= max[1]:
            max = (priorVal, post_priors[priorVal])
    return max
    
    

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


def getModelAccuracy(X, y, gaussianModels, priors):
    predictions = []
    for index, row in X.iterrows():
        prediction, predictionProb = predictX(gaussianModels, priors, row)
        predictions.append(prediction)

    predictSeries = pd.Series(predictions)
    return sum(predictSeries.values == y.values)/len(y)


def getFeatureImportances(Xtrain,ytrain,Xtest,ytest, npermutations = 3):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # find the original accuracy
    priors = computePriors(ytrain)
    gaussianModels = getFeatureGaussianModels(Xtrain, ytrain)       
    orig_accuracy = getModelAccuracy(Xtest, ytest, gaussianModels, priors)
    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtest2 = Xtest.copy()
            Xtest2[col] = Xtest[col].sample(frac=1, replace=False).values
            newAccuracy = getModelAccuracy(Xtest2, ytest, gaussianModels, priors)
            importances[col] += abs(orig_accuracy - newAccuracy)
        importances[col] = importances[col]/npermutations
    return importances


# testData = pd.read_csv('speeddating.csv')
# testData = testData[testData['attractive'] != '?']
# testData = testData[testData['sincere'] != '?']
# testData = testData[testData['intelligence'] != '?']
# testData = testData[testData['funny'] != '?']
# testData = testData[testData['ambition'] != '?']
# y = testData['decision_o']
# testData = testData[["attractive","sincere","intelligence","funny","ambition"]]


testData = pd.read_csv('titanic.csv')
testData = testData.dropna()
y = testData['Survived']
testData = testData[['Pclass', 'Age', 'Fare']]


Xtrain, ytrain, Xtest, ytest = train_test_split(testData, y, .5)
priors = computePriors(ytrain)
gaussianModels = getFeatureGaussianModels(Xtrain, ytrain)

print(getModelAccuracy(Xtest, ytest, gaussianModels, priors))

featureImportances = getFeatureImportances(Xtrain, ytrain, Xtest, ytest)
print(featureImportances)


