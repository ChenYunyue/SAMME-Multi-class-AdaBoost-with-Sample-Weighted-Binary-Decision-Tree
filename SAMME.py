"""Multi-class AdaBoost Classifier (SAMME) with sample weighted binary decision tree
When classifing, all the attribute values are 0 or 1!
"""

import math
import copy
import numpy as np

CLASS_NUM = 4

class SAMMEClassifier:
    """
    Multi-class Adaboost Classifier
    Training 50 weak classifiers (decision tree with depth of 2) with different sample weights
    """
    def __init__(self, weakLearnerNums=50):
        self.data = []
        self.target = []
        self.featureNums = 0
        self.datasetSize = 0
        self.dataWeights = []
        self.weakLearnerNums = weakLearnerNums
        self.learnerAlphas = []
        self.weakLearners = []

    def reset(self):
        self.data = []
        self.target = []
        self.featureNums = 0
        self.datasetSize = 0
        self.dataWeights = []
        self.learnerAlphas = []
        self.weakLearners = []
    
    def fit(self, data, target):
        """Train Adaboost"""
        self.data = np.array(data)
        self.target = np.array(target)
        self.datasetSize = self.data.shape[0]
        self.featureNums = self.data.shape[1]
        self.dataWeights = np.array([1/self.datasetSize for i in range(self.datasetSize)])
        self.weakLearners = [None for i in range(self.weakLearnerNums)]
        self.learnerAlphas = np.array([0 for i in range(self.weakLearnerNums)])
        # Train the weak learners one by one according to SAMME
        for m in range(self.weakLearnerNums):
            # Train a weak classifier (decision tree)
            tree = buildDecisionTree(self.data, self.target, parentTarget=self.target, dataWeights=self.dataWeights, parentWeights=self.dataWeights, maxDepth=2)
            self.weakLearners[m] = copy.deepcopy(tree)
            # Compute weighted error rate
            predicts = np.zeros(self.datasetSize)
            for i in range(self.datasetSize):
                predicts[i] = self.weakLearners[m].predict(data[i])
            weightSum = np.sum(self.dataWeights)
            errors = 0
            for i in range(self.datasetSize):
                if predicts[i] != target[i]:
                    errors = errors + self.dataWeights[i]
            err = errors / weightSum
            # Class number K = 4, compute alpha
            self.learnerAlphas[m] = max(0, math.log((1 - err) / (err + 1e-6))) + math.log(CLASS_NUM - 1)
            # Update data weights
            for i in range(self.datasetSize):
                if predicts[i] != target[i]:
                    self.dataWeights[i] = self.dataWeights[i] * math.exp(self.learnerAlphas[m])
            # Re-normalize weights
            weightSum = np.sum(self.dataWeights)
            self.dataWeights = self.dataWeights / weightSum

    def predict(self, data, legal=None):
        """Make prediction by weak classifiers"""
        predicts = np.zeros(4)
        for m in range(self.weakLearnerNums):
            predict = self.weakLearners[m].predict(data)
            predicts[predict] = predicts[predict] + self.learnerAlphas[m]
        # Select the best predicted class
        predict = predicts.argmax()
        return predict
