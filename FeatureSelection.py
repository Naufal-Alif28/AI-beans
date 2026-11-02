# REFERENCE: class resource "testing.py"
# IMPORT
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import os

# INPUT
data = "data_preparation/dataset_scaled.csv"
#sheet_name = "Dry_Beans_Dataset"
class_col = "Class"
output_dir = "feature_selection"
# hyperparameters
max_iter = 5000

# ALGORITHM
if __name__ == '__main__':
    #df = pd.read_excel(data,sheet_name=sheet_name)
    df = pd.read_csv(data)
    # define training dataset and classes
    train_x = df.drop(class_col, axis=1)
    y = df[class_col]
    # define model
    model = LogisticRegression(max_iter=max_iter)
    # fit model
    model_fit = model.fit(train_x, y)

    # calculate & show cross entropy
    prediction_probability = model_fit.predict_proba(train_x)
    cross_entropy = log_loss(y,prediction_probability)
    print(f"cross entropy : {cross_entropy}")
    # calculate and show training accuracy
    prediction_probability = model_fit.predict(train_x)
    training_accuracy = accuracy_score(y,prediction_probability)
    print(f"accuracy score : {training_accuracy}")

    # plot graph of coefficients
    classes = model.classes_
    features = train_x.columns.tolist()
    n_features = len(features)
    plt.figure(figsize=(n_features*0.6,7))
    for i,class_label in enumerate(classes):
        plt.plot(range(n_features), model.coef_[i], label=class_label)
    plt.xticks(list(range(n_features)), features, rotation=60)
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.grid(True)
    # save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/multinomial_LR_coefficients.pdf")
