# IMPORT
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math

# INPUT
data_raw = "datasets/Dry_Bean_Dataset.xlsx"
sheet_name = "Dry_Beans_Dataset"
class_col = "Class"
output_dir = "data_preparation"
seed = 92

# FUNCTIONS
def Reinsert(data,data_original,exclude:list):
    # reinsert excluded columns (features)
    for feature in exclude:
        data.insert(loc=len(data.columns), column=feature, value=data_original[feature]) 
    return data

def ShowClassProportion(data): # show value counts for each class and their proportion
    # calculate proportion
    count = data["Class"].value_counts()
    proportion = data["Class"].value_counts(normalize=True)
    stats = pd.DataFrame({
        "Count" : count,
        "Proportion(%)": proportion * 100
    })
    # show proportion
    print("Class stats:")
    print(stats)
    # visualise using pie chart
    plt.pie(stats["Count"], labels=stats.index, autopct='%1.1f%%')
    plt.savefig(f"{output_dir}/class_proportion.pdf")

def Scaling(data,exclude:list): # standardize features by removing the mean and scaling to unit variance
    # exclude Class
    df = data.drop(exclude,axis=1)
    # scale the dataframe
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df),columns=df.columns,index=df.index)
    # reinsert Class
    df = Reinsert(df,data,exclude)
    return df

def ShowStats(data,exclude:list,output_name:str): # calculate and show min, max, mean, stdev, and variance of each feature
    # exclude Class
    df = data.drop(exclude,axis=1)
    features = df.columns.tolist()
    # calculate sats of each feature
    stats = []
    for feature in features:
        stat = {}
        stat["Feature"] = feature
        stat["Min"] = df[feature].min()
        stat["Max"] = df[feature].max()
        stat["Mean"] = df[feature].mean()
        stat["Stdev"] = df[feature].std()
        stat["Var"] = math.sqrt(stat["Stdev"])
        stats.append(stat)
    # create dataframe and save as csv
    df = pd.DataFrame(stats)
    df.to_csv(f"{output_dir}/{output_name}.csv") 

def ShowBalance(data,class_col:str):
    counts = data[class_col].value_counts() # count instances in each class
    n = sum(counts) # total num of instances
    entropy = 0.0
    for c in counts: # calculate shannon entropy
        entropy += c / n * math.log(c / n)
    entropy = -entropy
    balance = entropy / math.log(len(counts))
    print(f"balance = {balance}") # print balance (0 -> perfect imbalance; 1 -> perfect balance)

# ALGORITHM
if __name__ == '__main__':
    # create output directory
    os.makedirs(output_dir,exist_ok=True)
    # read xlsx, store in dataframe
    df_unscaled = pd.read_excel(data_raw, sheet_name=sheet_name)
    # show class proportions
    ShowClassProportion(df_unscaled)
    # scale dataset
    df_scaled = Scaling(df_unscaled,["Class"])
    # save dataframe to csv
    df_scaled.to_csv(f"{output_dir}/dataset_scaled.csv")
    # calculate stats of features of both unscaled and scaled dataset
    ShowStats(df_unscaled,exclude=["Class"], output_name="unscaled_train_stats")
    ShowStats(df_scaled,exclude=["Class"], output_name="scaled_train_stats")
    # calculate class balance measure based on shannon entropy
    ShowBalance(df_scaled, class_col=class_col)