import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[1:5]
y = iris.target

CLASS_NUM = 4

"""sample weighted numerical threshold binary decision tree classifier
Input:
    - Data:
        - batch data for training of shape (n samples * d dimensions)
        - single or batch data of shape (d dimensions) or (n samples * d dimensions) for prediction
    - Target: array of labels of shape (n samples)
        - NB: support multi-class, but classes must be coded as {0, 1, 2, ...}
Output:
    - Prediction: predicted label array of shape (n samples)
Methods:
    - train: build a decision tree from training data and target
    - predict:
        - predict data by the trained decision tree
        - the decision tree must be trained first
"""

class TreeNode:
    """threshold decision tree node"""
    def __init__(self, attribute=None, leftBranch=None, rightBranch=None, result=None):
        self.attribute = attribute
        self.threshold = None
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.result = result
    
    def predict(self, data):
        """support batch input"""
        def runTree(data):
            if self.result != None:
                return self.result
            else:
                newData = np.delete(data, self.attribute)
                if data[self.attribute] == 0:
                    return self.leftBranch.predict(newData)
                else:
                    return self.rightBranch.predict(newData)

        data = np.array(data)
        # single input
        if np.size(data.shape) == 1:
            return runTree(data)
        # batch inputs
        else:
            batchSize = data.shape[0]
            results = np.zeros(batchSize)
            for ind, sample in enumerate(data):
                results[ind] = runTree(sample)
            return results
        
# class DecisionTreeClassifier:
#     """sample weighted numerical threshold binary decision tree classifier"""
#     def __init__(self):
#         self.root

def dataSplit(data, target, dataWeights, attribute, threshold):
    """Split the data, target, and weights by attribute"""
    leftData = data[data[:, attribute] <= threshold]
    leftData = np.delete(leftData, attribute, 1)
    leftTarget = target[data[:, attribute] <= threshold]
    leftDataWeights = dataWeights[data[:, attribute] <= threshold]

    rightData = data[data[:, attribute] > threshold]
    rightData = np.delete(rightData, attribute, 1)
    rightTarget = target[data[:, attribute] > threshold]
    rightDataWeights = dataWeights[data[:, attribute] > threshold]
    
    return (leftData, leftTarget, leftDataWeights, rightData, rightTarget, rightDataWeights)

    # zeroData = data[data[:,attribute]==0]
    # zeroData = np.delete(zeroData, attribute, 1)
    # zeroTarget = target[data[:,attribute]==0]
    # zeroDataWeights = dataWeights[data[:,attribute]==0]
    # oneData = data[data[:,attribute]==1]
    # oneData = np.delete(oneData, attribute, 1)
    # oneTarget = target[data[:, attribute]==1]
    # oneDataWeights = dataWeights[data[:,attribute]==1]
    # return (zeroData, zeroTarget, zeroDataWeights, oneData, oneTarget, oneDataWeights)


def gini(target, dataWeights):
    """Weighted Gini Impurity"""
    # empty set: gini = 1
    if target.shape[0] == 0:
        return 1
    pList = np.zeros(CLASS_NUM)
    # weighted sum
    for i in range(target.shape[0]):
        pList[target[i]] += dataWeights[i]
    # normalize probability
    weightSum = np.sum(dataWeights)
    pList = pList / weightSum
    impurity = 1 - sum(pList*pList)
    return impurity


def pluralityValue(target, dataWeights):
    """Majority vote"""
    votes = np.zeros(CLASS_NUM)
    # vote based on weights
    for i in range(target.shape[0]):
        votes[target[i]] += dataWeights[i]
    majority = np.argmax(votes)
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
        