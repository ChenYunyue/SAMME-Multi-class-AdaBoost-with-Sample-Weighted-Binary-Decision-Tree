import numpy as np

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
    def __init__(self, attribute=None, leftBranch=None, rightBranch=None, result=None, threshold=None):
        self.attribute = attribute
        self.threshold = threshold
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
                if data[self.attribute] <= self.threshold:
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
        
class MyDecisionTreeClassifier:
    """sample weighted numerical threshold binary decision tree classifier"""
    def __init__(self):
        self.root = None
        self.classNumber = -1

    def dataSplit(self, data, target, dataWeights, attribute, threshold):
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

    def gini(self, target, dataWeights):
        """Weighted Gini Impurity"""
        # empty set: gini = 1
        if target.shape[0] == 0:
            return 1
        pList = np.zeros(self.classNumber)
        # weighted sum
        for i in range(target.shape[0]):
            pList[target[i]] += dataWeights[i]
        # normalize probability
        weightSum = np.sum(dataWeights)
        pList = pList / weightSum
        impurity = 1 - sum(pList * pList)
        return impurity

    def pluralityValue(self, target, dataWeights):
        """Majority vote"""
        votes = np.zeros(self.classNumber)
        # vote based on weights
        for i in range(target.shape[0]):
            votes[target[i]] += dataWeights[i]
        majority = np.argmax(votes)
        return majority

    def buildDecisionTree(self, data, target, parentTarget, dataWeights, parentWeights, maxDepth=50, currentDepth=0):
        """Build decision tree recursively, using gini impurity"""
        # data is empty
        if target.shape[0] == 0:
            result = self.pluralityValue(parentTarget, parentWeights)
            return TreeNode(result=result)
        # all data have the same class
        if (target == target[0]).all() == True:
            result = target[0]
            return TreeNode(result=result)
        featureNums = data.shape[1]
        # attributes is empty
        if featureNums == 0:
            result = self.pluralityValue(target, dataWeights)
            return TreeNode(result=result)
        # reached max depth
        if currentDepth >= maxDepth:
            result = self.pluralityValue(target, dataWeights)
            return TreeNode(result=result)
        # continue to expand the tree
        bestGiniDiff = 0
        bestAttribute = -1
        unsplitGini = self.gini(target, dataWeights)
        totalWeightSum = np.sum(dataWeights)
        # find the best attribute
        for attribute in range(featureNums):
            # travers all the numerical values
            for threshold in data[:, attribute]:
                (leftData, leftTarget, leftDataWeights, rightData, rightTarget, rightDataWeights) = self.dataSplit(data, target, dataWeights, attribute, threshold)
                leftGini = self.gini(leftTarget, leftDataWeights)
                rightGini = self.gini(rightTarget, rightDataWeights)
                leftWeightSum = np.sum(leftDataWeights)
                rightWeightSum = np.sum(rightDataWeights)
                # compare gini impurity before and after the split
                weightedGini = (leftWeightSum/totalWeightSum) * leftGini + (rightWeightSum/totalWeightSum) * rightGini
                giniDiff = unsplitGini - weightedGini
                if giniDiff > bestGiniDiff:
                    bestGiniDiff = giniDiff
                    bestAttribute = attribute
                    bestThreshold = threshold
        # if such split is useful, split the dataset and expand the tree
        if bestGiniDiff > 0:
            (leftData, leftTarget, leftDataWeights, rightData, rightTarget, rightDataWeights) = self.dataSplit(data, target, dataWeights, bestAttribute, bestThreshold)
            leftBranch = self.buildDecisionTree(leftData, leftTarget, target, leftDataWeights, dataWeights, maxDepth=maxDepth, currentDepth=currentDepth+1)
            rightBranch = self.buildDecisionTree(rightData, rightTarget, target, rightDataWeights, dataWeights, maxDepth=maxDepth, currentDepth=currentDepth+1)
            return TreeNode(attribute=bestAttribute, leftBranch=leftBranch, rightBranch=rightBranch, threshold=bestThreshold)
        # if such split is not useful, just return the majority vote
        else:
            result = self.pluralityValue(target, dataWeights)
            return TreeNode(result=result)

    def train(self, data, target, dataWeights=None, maxDepth=50):
        if dataWeights == None:
            dataWeights = np.ones(data.shape[0])
        labels = np.unique(target)
        self.classNumber = labels.shape[0]
        self.root = self.buildDecisionTree(data, target, target, dataWeights, dataWeights, maxDepth)
    
    def predict(self, data):
        if self.root == None:
            print("Please train the decision tree classifier FIRST!")
        else:
            results = self.root.predict(data)
            return results
        