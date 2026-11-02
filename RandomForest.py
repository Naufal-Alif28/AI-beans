# iMPORT
import pandas as pd
import os
from FinalProjectLib import SetSeed,LoadData,CrossValidation
from sklearn.ensemble import RandomForestClassifier

# INPUT
data = "data_preparation/dataset_unscaled.csv"
class_col = "Class"
output_dir = "results"
output_file = "cv_RF_unscaled_1.csv"
# random forest parameters
n_estimators = 300 # number of trees in the forest
max_depth =  30 # maximum tree depth, until leaf pure if None
max_features = "sqrt" # maximum number of feature samled in each bootstrap (sqrt mean sqrt(n_feature))
max_leaf = None # max number of leaf
max_samples = 0.9 # maximum sampling ratio per bootstrap
min_samples_leaf = 10 # minimum samples per leaf node
min_samples_split = 20 # minimum sample for node splitting
class_weight = "balanced" # class weighting option in case of class size imbalance ("balanced" for automatic balancing)
# cross-validation parameters
seed = 92 # (!) KEEP CONSISTENT ACROSS VALIDATION RUNS (!) seed int for random_state
n_splits = 10 # number of splits for StratifiedKFold n_splits
scoring_metrics  = ["accuracy","precision_macro","recall_macro","f1_macro"]

# ALGORITHM
if __name__ == '__main__':
     # set seed
    SetSeed(seed)

    # create output file and check output validity
    os.makedirs(output_dir,exist_ok=True)
    f = open(f"{output_dir}/{output_file}",mode='w')
    f.close()

    # define the cross-validation data (x) and label (y)
    x,y,y_map = LoadData(data,class_col=class_col)
    # define the model
    model = RandomForestClassifier(
        n_estimators = n_estimators,
        criterion = "gini",
        max_depth = max_depth,
        max_features = max_features,
        max_leaf_nodes = max_leaf,
        bootstrap = True,
        random_state = seed,
        class_weight = class_weight,
        max_samples = max_samples,
        min_samples_leaf = min_samples_leaf,
        min_samples_split = min_samples_split
    )

    # do cross validation
    print("doing cross-validation...")
    results = CrossValidation(model,x,y,scoring_metrics,seed,n_splits)
    # save cv result
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/{output_file}")
    print("cross-validation results has been saved")