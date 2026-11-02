# REFERENCE
# class resource "NeuralNetwork.py"
# reference article Koklu,M. & Ozkan, I.A. (2020)

# IMPORT
import torch
import torch.nn as nn
import torch.optim
import pandas as pd
from skorch import NeuralNetClassifier
import os
from FinalProjectLib import SetSeed,LoadData,CrossValidation,ComputeWeight

# INPUT
# INPUT
# INPUT
data = "data_preparation/dataset_scaled.csv"
class_col = "Class"
output_dir = "results"
output_file = "cv_NN_scaled_1.csv"
# neural network parameters
input_size = 16
h1_size = 32 # i-th hidden layer size
h2_size = 32
h3_size = 16
n_class = 7
n_epoch = 75
batch_size = 32
learning_rate = 0.001
dropout_1 = 0.2 # dropout ratio
dropout_2 = 0.2
dropout_3 = 0.2
# cross validation parameters
seed = 92 # (!) KEEP CONSISTENT ACROSS VALIDATION RUNS (!) seed int for random_state
n_splits = 10 # number of splits for StratifiedKFold n_splits
scoring_metrics = ["accuracy","precision_macro","recall_macro","f1_macro"]

# CLASS
class NN(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, h3_size, dropout_1, dropout_2, dropout_3, n_class): # neurl network construction
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size, h1_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_1)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_2)
        self.fc3 = nn.Linear(h2_size, h3_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_3)
        self.fc4 = nn.Linear(h3_size, n_class)
    def forward(self,x): # forward pass
        h1 = self.dropout1(self.relu1(self.fc1(x)))
        h2 = self.dropout2(self.relu2(self.fc2(h1)))
        h3 = self.dropout3(self.relu3(self.fc3(h2)))
        out = self.fc4(h3)
        return out

# ALGORITHM
if __name__ == '__main__':
    # set seed
    SetSeed(seed)

    # create output file and check output validity
    os.makedirs(output_dir, exist_ok=True)
    f = open(f"{output_dir}/{output_file}", mode='w')
    f.close()

    # define the cross-validation data (x) and label (y)
    x,y,y_map = LoadData(data, class_col=class_col)
    # compute class weight
    class_weight = ComputeWeight(y)
    # define the model (using skorch wrapper)
    model = NeuralNetClassifier(
        NN,
        module__input_size = input_size,
        module__h1_size = h1_size,
        module__dropout_1 = dropout_1,
        module__h2_size = h2_size,
        module__dropout_2 = dropout_2,
        module__h3_size = h3_size,
        module__dropout_3 = dropout_3,
        module__n_class = n_class,
        optimizer = torch.optim.Adam,
        lr = learning_rate,
        batch_size = batch_size,
        max_epochs = n_epoch,
        criterion = nn.CrossEntropyLoss,
        criterion__weight = class_weight
    )
    # do cross-validation
    print("doing cross-validation...")
    results = CrossValidation(model, x, y, scoring_metrics, seed,n_splits)
    # save cv result
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/{output_file}")
    print("cross-validation results has been saved")