# library I made for functions that will be used
# in common for all models in the project

def SetSeed(seed:int, what:set={"numpy","random","torch"}):
    import random
    import numpy as np
    import torch

    # set seed for seed in random, numpy, and pytorch (for consistency)
    if "random" in what:
        random.seed(seed)
    if "numpy" in what:
        np.random.seed(seed)
    if "torch" in what:
        torch.manual_seed(seed)

def LoadData(data_path:str, class_col:str, features:list=None, y_map:dict=None):
    import pandas as pd
    import numpy as np

    # read data (csv) to dataframe
    df = pd.read_csv(data_path,index_col=False)
    # if features not specified, use all column except class_col
    # might be used if only using specified features or RFE
    if features == None:
        features_list = df.columns.tolist()
        features_list.remove(class_col)
    else:
        features_list = features
    # divide data to training dataset (x) and classes/label (y)
    x = df[features_list].to_numpy(dtype=np.float32)
    classes = list(set(df[class_col])) # nonredundant list of classes
    if y_map == None:
        y_map = {j : i for i,j in enumerate(classes)} # map for converting classes to int
    print(y_map)
    y = np.array([y_map[j] for j in df[class_col]]) # y in int format
    return x, y, y_map

def CrossValidation(model, x, y, scoring_metrics:list, seed, n_splits):
    from sklearn.model_selection import StratifiedKFold,cross_validate

    # set cross-validation splitter using Stratified K-Fold method
    cv =  StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    # do cross-validation
    scores = cross_validate(model, x, y, scoring=scoring_metrics, cv=cv, return_train_score=True)
    return scores

def ComputeWeight(y): # compute weight for each class to balance the dataset
    from sklearn.utils.class_weight import compute_class_weight
    import torch
    import numpy as np

    classes = list(set(y)) # nonredundant list of classes
    classes = np.array(classes) # covert to numpy array
    weight = compute_class_weight(class_weight="balanced", classes=classes,y=y) # compute weight
    weight = torch.tensor(weight, dtype=torch.float32) # convert to pytorch tensor
    return weight