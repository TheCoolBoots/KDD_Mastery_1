import unittest
import pandas as pd
import numpy as np
from GaussianBayesianClassifier import *

testDFData = {'numPastGirlfriends': [1,3,1,4],
                'salary':[1000, 304151, 25134, 15124],
                'numFingers':[10, 10, 10, 9]}

targetData = [0,1,0,1]

testDF = pd.DataFrame(testDFData)
targetSeries = pd.Series(targetData, name='hasGirlfriend')  

class testGBC(unittest.TestCase):

    def test_ComputePriors(self):
        priors = computePriors(targetSeries)
        expected = {'hasGirlfriend=0':.5,'hasGirlfriend=1':.5}
        self.assertEqual(expected, priors)

    def test_GaussianLikelihood(self):
        gaussianModels = getFeatureGaussianModels(testDF, targetSeries)
        actual = getGaussianLikelihood(159637.5, 159637.5, 204372.95164600428)
        expected = 0
        print(actual)

    def test_FeatureGaussianModels(self):
        actual = getFeatureGaussianModels(testDF, targetSeries)
        expected = {
            'numPastGirlfriends': {0: (1.0, 0.0), 1: (3.5, 0.7071067811865476)}, 
            'salary': {0: (13067.0, 17065.31505715614), 1: (159637.5, 204372.95164600428)},
            'numFingers': {0: (10.0, 0.0), 1: (9.5, 0.7071067811865476)}
        }
        self.assertAlmostEqual(actual,expected)

    # def test_Posteriors(self):
    #     pass


if __name__ == '__main__':
    unittest.main()