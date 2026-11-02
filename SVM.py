# iMPORT
import pandas as pd
import os
from FinalProjectLib import SetSeed,LoadData,CrossValidation
from sklearn.svm import SVC

# INPUT
# INPUT
data = "data_preparation/dataset_scaled.csv"
class_col = "Class"
output_dir = "results"
output_file = "cv_SVM_scaled_1.csv"
# SVM parameters
kernel = "rbf" # kernel type for SVM ("poly","linear","rbf","sigmoid","precomputed")
degree = 3 # polynomoial degree
gamma = "scale" # gamma coefficient for rbf or polynomial
class_weight = None # class weighting option in case of class size imbalance ("balanced" for automatic balancing)
max_iter = 2000 # maximum number of iterations
func_shape = "ovo" # decision function shape ("ovo":one-vs-one;"ovr":"one-vs-rest")
# cross-validation parameters
seed = 92 # (!) KEEP CONSISTENT ACROSS VALIDATION RUNS (!) seed int for random_state
n_splits = 10 # number of splits for StratifiedKFold n_splits
scoring_metrics  = ["accuracy","precision_macro","recall_macro","f1_macro"]
features = ["Area","Perimeter","MajorAxisLength","MinorAxisLength","AspectRation","Eccentricity","ConvexArea","EquivDiameter","Extent","roundness","Compactness","ShapeFactor1","ShapeFactor2","ShapeFactor3","ShapeFactor4"]

# ALGORITHM
if __name__ == '__main__':
     # set seed
    SetSeed(seed)

    # create output file and check output validity
    os.makedirs(output_dir,exist_ok=True)
    f = open(f"{output_dir}/{output_file}",mode='w')
    f.close()

    # define the cross-validation data (x) and label (y)
    x,y,y_map = LoadData(data,class_col=class_col,features=features)
    # define the model
    model = SVC(
        kernel = kernel,
        class_weight = class_weight,
        max_iter = max_iter,
        decision_function_shape = func_shape,
        random_state = seed
        #degree = degree
    )

    # do cross validation
    print("doing cross-validation...")
    results = CrossValidation(model,x,y,scoring_metrics,seed,n_splits)
    # save cv result
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/{output_file}")
    print("cross-validation results has been saved")