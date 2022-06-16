# SAMME-Multi-class-AdaBoost-with-Sample-Weighted-Binary-Decision-Tree

Reference:
1. Hastie, T., Rosset, S., Zhu, J., & Zou, H. (2009). Multi-class adaboost. Statistics and its Interface, 2(3), 349-360.
2. https://github.com/Hanbo-Sun/Multiclass_AdaBoost
3. https://github.com/gengjia0214/Python-Multiclass-AdaBoost-SAMME
4. https://github.com/lucksd356/DecisionTrees

SAMME and decision tree can work independently.

## Numerical Sample Weighted Binary Decision Tree

* Input:
  * Data:
    * Batch data for training of shape `(n samples, d dimensions)`;
    * Single or batch data of shape `(d dimensions)` or `(n samples, d dimensions)` for prediction.
  * Target:
    * Array of labels of shape `(n samples)`;
    * *NB: Support multi-class, but class labels must be coded as* `{0,1,2,...}`.
* Output:
  * Prediction: Predicted label array of shape `(n samples)`.
* Methods:
  * `train()`: Build a decision tree from training data and target.
  * `predict()`:
    * Predict data by the trained decision tree;
    * *NB: The decision tree must be trained first.*