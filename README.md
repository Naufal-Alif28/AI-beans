## Neural Network, SVM, and Rendom Forest Models Comparison in Multiclass Classification of Dry Beans

Part of the final project of "Fundamentals of Artificial Intelligence" course (Kyoto University International Education Program, Fall 2025)

Overall, the programs run as:
1. `DataPreparation.py` 
    - `ShowClassProportion` calculate and visualise the proportion of each class 
    - `Scaling` standardises the features by scaling to unit variance
    - `ShowStats` summarises basic statistics of the dataset
    - `ShowBalance` calculate the classes balance by calculating the entropy

2. `FeatureSelection.py` does a quick, iterative logistic regression to estimate the contribution of each feature to the class distribution

3. `OutlierDetection.py`
    - `VisualiseUnivariateDist` visualise the data distribution in each feature (univariate)
    - `CalculateMahalanobis` and `VisualiseMahalanobis` calculates the Mahalanobis distance of the data and visualise it with a 2-d PCA plot while highlighting outlier data points

4. `NeuralNetwork.py`, `RandomForest.py`, and `SVM.py` each does training and testing of the respective model using `scikit-learn`'s stratified k-fold method with a common static seed. Neural network were constructed on `pytorch`, wrapped with `skorch`
