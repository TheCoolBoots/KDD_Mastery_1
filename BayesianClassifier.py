import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts

def compute_priors(y):
    valueCounts = y.value_counts().sort_index()
    priors = {f'{y.name}={item[0]}': item[1]/valueCounts.sum() for item in valueCounts.iteritems()}
    return priors

def specific_class_conditional(x,xv,y,yv):
    # P(mandarin | width=8.4) = P(mandarin and width=8.4)/P(width = 8.4)
    frame = x.to_frame()
    frame[y.name] = y
    # print(frame)
    yvFrame = frame[frame[y.name] == yv]
    # print(yvFrame)
    xvyvFrame = yvFrame[yvFrame[x.name] == xv]
    # print(xvyvFrame)
    Pxvyv = len(xvyvFrame)/len(frame)
    Pyv = len(yvFrame)/len(frame)
    return Pxvyv/Pyv
    # 0.14754098360655737

def class_conditional(X,y):
    output = {}
    for col in X.columns:
        # print(col)
        for valx in X[col].unique():
            # print(valx)
            for valy in y.unique():
                # print(valy)
                output[f'{col}={valx}|{y.name}={valy}'] = specific_class_conditional(X[col], valx, y, valy)

    return output

def posteriors(probs,priors,x):
    
    # probability P(A|B) = P(B|A)P(A)/(P(B|A)*P(A) + sum(P(B|!A) * P(!A)))
    
    # input = 
    '''
    Pclass         3
    Sex       female
    Age           20
    '''
    
#     numerator1 = priors['Survived=0'] * probs['Pclass=3|Survived=0']* * probs['Sex=female|Survived=0'] * probs['Age=20|Survived=0']
#     numerator2 = priors['Survived=1'] * probs['Pclass=3|Survived=1'] * probs['Sex=female|Survived=1'] * probs['Age=20|Survived=1']

#     denominator = numerator1+numerator2
    
#     result = numerator1/denominator

    numerators = {}
    for i, prior in priors.items():
        numerator = priors[i]
        for j, val in x.iteritems():
            probIndex = f'{j}={val}|{i}'
            if probIndex in probs:
                numerator *= probs[probIndex]
            else:
                numerator = .5
                break
        numerators[i] = numerator
        
    denominator = sum(numerators.values())
    post_priors = {}
    for i, prior in priors.items():
        indexStr = None
        for j, val in x.iteritems():
            if indexStr is None:
                indexStr = f'{i}|{j}={val}'
            else:
                indexStr += f',{j}={val}'
        post_priors[indexStr] = numerators[i]/denominator
        
    return post_priors
            

# {'Survived=0|Pclass=3,Sex=female,Age=20': 0.46699312907215196,
#  'Survived=1|Pclass=3,Sex=female,Age=20': 0.533006870927848}

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
    
    
    # can't use sklearn b/c solution uses numpy seed for random ordering
    # Xtrain,Xtest,ytrain,ytest = tts(X, y, test_size=test_frac, shuffle=True)
    return Xtrain,ytrain,Xtest,ytest

def exercise_6(Xtrain,ytrain,Xtest,ytest):
    # use xtrain, ytrain to generate probs and priors
    probs = class_conditional(Xtrain, ytrain)
    priors = compute_priors(ytrain)
    
    expectedVals = []
    
    # print(probs)
    for index, row in Xtest.iterrows():
        expectedValDict = posteriors(probs, priors, row)
        likelyCond, highestProb = None, 0
        for cond, prob in expectedValDict.items():
            if prob > highestProb:
                likelyCond = cond
                highestProb = prob
        condVal = int(likelyCond.split('|')[0].split('=')[-1])
        expectedVals.append(condVal)
    
    expectedVals = pd.Series(expectedVals, name='Survived')
    
    equal = expectedVals.values == ytest.values
    
    return sum(equal)/len(ytest)

def exercise_7(Xtrain,ytrain,Xtest,ytest, npermutations = 10):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # find the original accuracy
    orig_accuracy = exercise_6(Xtrain,ytrain,Xtest,ytest)
    print(orig_accuracy)
    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtest2 = Xtest.copy()
            Xtest2[col] = Xtest[col].sample(frac=1, replace=False).values
            newAccuracy = exercise_6(Xtrain,ytrain,Xtest2,ytest)
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

# testData = testData.apply(lambda row: row.astype(int))

# pd.cut(fruits.width.loc[fruits.fruit_name=='orange'],5,retbins=True)

titanic_df = pd.read_csv('titanic.csv')

# titanic_df = pd.DataFrame()

features = ['Pclass','Survived','Fare','Age']
titanic_df = titanic_df[features]
titanic_df['Pclass'] = titanic_df['Pclass'].fillna(titanic_df['Pclass'].mode()).astype(int)
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
titanic_df['Age'] = (titanic_df['Age']/10).astype(str).str[0].astype(int)*10
titanic_df['Fare'] = pd.cut(titanic_df['Fare'], 8)

print(titanic_df)
# titanic_df = titanic_df.dropna()

Xtrain, ytrain, Xtest, ytest = train_test_split(titanic_df[['Pclass', 'Fare', 'Age']], titanic_df['Survived'], .3)

importances = exercise_7(Xtrain, ytrain, Xtest, ytest)
print(importances)