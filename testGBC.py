from random import gauss
import unittest
import pandas as pd
import numpy as np
from GaussianBayesianClassifier import *

trainXData = {'numPastGirlfriends': [1,3,1,4],
                'salary':[1000, 304151, 25134, 15124],
                'numFingers':[10, 10, 10, 9]}

trainYData = [0,1,0,1]

trainXDF = pd.DataFrame(trainXData)
trainYDF = pd.Series(trainYData, name='hasGirlfriend')  


testXData = [3,304151,10]
testXSeries = pd.Series(testXData, index=['numPastGirlfriends', 'salary', 'numFingers'])

class testGBC(unittest.TestCase):

    def test_ComputePriors(self):
        priors = computePriors(trainYDF)
        expected = {'hasGirlfriend=0':.5,'hasGirlfriend=1':.5}
        self.assertEqual(expected, priors)

    def test_GaussianLikelihood(self):
        gaussianModels = getFeatureGaussianModels(trainXDF, trainYDF)
        actual = getGaussianProbability(0, 0, 1)
        expected = .07965567455405798
        self.assertAlmostEqual(actual, expected)

    def test_FeatureGaussianModels(self):
        actual = getFeatureGaussianModels(trainXDF, trainYDF)
        expected = {
            'numPastGirlfriends': {0: (1.0, 0.0), 1: (3.5, 0.7071067811865476)}, 
            'salary': {0: (13067.0, 17065.31505715614), 1: (159637.5, 204372.95164600428)},
            'numFingers': {0: (10.0, 0.0), 1: (9.5, 0.7071067811865476)}
        }
        self.assertAlmostEqual(actual,expected)

    def test_Posteriors(self):
        guassianModels = getFeatureGaussianModels(trainXDF, trainYDF)
        priors = computePriors(trainYDF)
        actual = posteriors(guassianModels, priors, testXSeries)


if __name__ == '__main__':
    unittest.main()