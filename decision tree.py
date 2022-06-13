import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[1:5]
y = iris.target

print(X)



"""Sample weighted numerical threshold binary decision tree
"""
class TreeNode:
    """Decision Tree Node"""
    def __init__(self, attribute=-1, leftBranch=None, rightBranch=None, result=None):
        self.attribute = attribute
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.result = result
    
    def predict(self, data):
        """Do no support batch input"""
        if self.result != None:
            return self.result
        else:
            newData = np.delete(data, self.attribute)
            if data[self.attribute] == 0:
                return self.leftBranch.predict(newData)
            else:
                return self.rightBranch.predict(newData)


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
        rightBranch = buildDecisionTree(oneData, oneTarget, target, oneDataWeights, dataWeights, maxDepth=maxDepth, currentDepth=currentDepth+1)
        return TreeNode(attribute=bestAttribute, zeroBranch=zeroBranch, rightBranch=rightBranch)
    # if such split is not useful, just return the majority vote
    else:
        result = pluralityValue(target, dataWeights)
        return TreeNode(result=result)
        