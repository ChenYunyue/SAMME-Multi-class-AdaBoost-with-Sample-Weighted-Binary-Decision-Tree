"""Multi-class AdaBoost Classifier (SAMME) with sample weighted binary decision tree

Reference:
[1] Hastie, T., Rosset, S., Zhu, J., & Zou, H. (2009). Multi-class adaboost. Statistics and its Interface, 2(3), 349-360.
[2] https://github.com/Hanbo-Sun/Multiclass_AdaBoost
[3] https://github.com/gengjia0214/Python-Multiclass-AdaBoost-SAMME
[4] https://github.com/lucksd356/DecisionTrees

When classifing, all the attribute values are 0 or 1!
To do: Adapt the binary decision tree to numerical values.
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


"""Sample weighted 0-1 value binary decision tree
"""
class TreeNode:
    """Decision Tree Node"""
    def __init__(self, attribute=-1, zeroBranch=None, oneBranch=None, result=None):
        self.attribute = attribute
        self.zeroBranch = zeroBranch
        self.oneBranch = oneBranch
        self.result = result
    
    def predict(self, data):
        """Do no support batch input"""
        if self.result != None:
            return self.result
        else:
            newData = np.delete(data, self.attribute)
            if data[self.attribute] == 0:
                return self.zeroBranch.predict(newData)
            else:
                return self.oneBranch.predict(newData)


def dataSplit(data, target, dataWeights, attribute):
    """Split the data, target, and weights by attribute"""
    zeroData = data[data[:,attribute]==0]
    zeroData = np.delete(zeroData, attribute, 1)
    zeroTarget = target[data[:,attribute]==0]
    zeroDataWeights = dataWeights[data[:,attribute]==0]
    oneData = data[data[:,attribute]==1]
    oneData = np.delete(oneData, attribute, 1)
    oneTarget = target[data[:, attribute]==1]
    oneDataWeights = dataWeights[data[:,attribute]==1]
    return (zeroData, zeroTarget, zeroDataWeights, oneData, oneTarget, oneDataWeights)


def gini(target, dataWeights):
    """Weighted Gini Impurity"""
    # empty set: gini = 1
    if target.shape[0] == 0:
        return 1
    pList = np.zeros(CLASS_NUM)
    # weighted sum
    for i in range(target.shape[0]):
        pList[target[i]] = pList[target[i]] + dataWeights[i]
    # normalize probability
    weightSum = np.sum(dataWeights)
    pList = pList / weightSum
    impurity = 1 - sum(pList*pList)
    return impurity


def pluralityValue(target, dataWeights):
    """Majority vote"""
    results = np.zeros(CLASS_NUM)
    # vote based on weights
    for i in range(target.shape[0]):
        results[target[i]] = results[target[i]] + dataWeights[i]
    majority = results.argmax()
    return majority


def buildDecisionTree(data, target, parentTarget, dataWeights, parentWeights, maxDepth=2, currentDepth=0):
    """Build decision tree recursively, using gini impurity"""
    # data is empty
    if target.shape[0] == 0:
        result = pluralityValue(parentTarget, parentWeights)
        return TreeNode(result=result)
    # all data have the same class
    if (target == target[0]).all() == True:
        result = target[0]
        return TreeNode(result=result)
    featureNums = data.shape[1]
    # attributes is empty
    if featureNums == 0:
        result = pluralityValue(target, dataWeights)
        return TreeNode(result=result)
    # reached max depth
    if currentDepth >= maxDepth:
        result = pluralityValue(target, dataWeights)
        return TreeNode(result=result)
    # continue to expand the tree
    bestGiniDiff = 0
    bestAttribute = -1
    unsplitGini = gini(target, dataWeights)
    totalWeightSum = np.sum(dataWeights)
    # find the best attribute
    for attribute in range(featureNums):
        (zeroData, zeroTarget, zeroDataWeights, oneData, oneTarget, oneDataWeights) = dataSplit(data, target, dataWeights, attribute)
        zeroGini = gini(zeroTarget, zeroDataWeights)
        oneGini = gini(oneTarget, oneDataWeights)
        zeroWeightSum = np.sum(zeroDataWeights)
        oneWeightSum = np.sum(oneDataWeights)
        # compare gini impurity before and after the split
        weightedGini = (zeroWeightSum/totalWeightSum)*zeroGini + (oneWeightSum/totalWeightSum)*oneGini
        giniDiff = unsplitGini - weightedGini
        if giniDiff > bestGiniDiff:
            bestGiniDiff = giniDiff
            bestAttribute = attribute
    # if such split is useful, split the dataset and expand the tree
    if bestGiniDiff > 0:
        (zeroData, zeroTarget, zeroDataWeights, oneData, oneTarget, oneDataWeights) = dataSplit(data, target, dataWeights, bestAttribute)
        zeroBranch = buildDecisionTree(zeroData, zeroTarget, target, zeroDataWeights, dataWeights, maxDepth=maxDepth, currentDepth=currentDepth+1)
        oneBranch = buildDecisionTree(oneData, oneTarget, target, oneDataWeights, dataWeights, maxDepth=maxDepth, currentDepth=currentDepth+1)
        return TreeNode(attribute=bestAttribute, zeroBranch=zeroBranch, oneBranch=oneBranch)
    # if such split is not useful, just return the majority vote
    else:
        result = pluralityValue(target, dataWeights)
        return TreeNode(result=result)
        