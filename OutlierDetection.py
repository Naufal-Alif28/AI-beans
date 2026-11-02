# REFERENCE: class resource "data_collection.py", "unsupervised.py"
# IMPORT
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# INPUT
data_raw = "datasets/Dry_Bean_Dataset.xlsx"
sheet_name = "Dry_Beans_Dataset"
output_dir = "outlier_detection"
chi_square = 26.296

# FUNCTIONS
def VisualiseUnivariateDist(data,exclude): 
    # exclude : excluded feature name (list,str)
    features = data.columns.to_numpy()
    features = features[~np.isin(features,exclude)]
    # plotting
    # determine optimal subplots dimension
    l = len(features)
    x,y = 1,0
    while x*y < l:
        if x > y:
            y += 1
        else:
            x += 1
    # subplot axes for each features
    fig,axes = plt.subplots(x,y,figsize=(x*5,y*5)) 
    # create barplot for each feature
    coordinates = [(i,j) for i in range(x) for j in range(y)]
    for k in range(l):
        i,j = coordinates[k]
        axes[i][j].boxplot(data[features[k]],orientation="horizontal")
        axes[i][j].set_title(features[k])
    fig.suptitle("Univariate Distribution : barplot")
    # save and show barplot
    plt.savefig(f"{output_dir}/univariate_barplot.pdf")
    # create histogram for each feature
    for k in range(l):
        i,j = coordinates[k]
        axes[i][j].hist(data[features[k]])
        axes[i][j].set_title(features[k])
    fig.suptitle("Univariate Distribution : histogram")
    # save histogram
    plt.savefig(f"{output_dir}/univariate_histogram.pdf")

def CalculateMahalanobis(data,exclude):
    # exclude : excluded feature name (list,str)
    df = data.drop(exclude,axis=1)
    # calculate mean vector
    mean_vector = df.mean().values
    # calculate inverse covarience matrix
    covarience_matrix = np.cov(df.T)
    inverse_covariance_matrix = np.linalg.inv(covarience_matrix)
    # create list of Mahalanobis distances
    mahalanobis_distances = []
    for i in range(len(df)):
        point = df.iloc[i].values
        distance = mahalanobis(point,mean_vector,inverse_covariance_matrix)
        mahalanobis_distances.append(distance)
    # reinsert excluded columns (features)
    for feature in exclude:
        df.insert(loc=len(df.columns),column=feature,value=data[feature]) 
    # insert calculated mahalanobis distance
    df.insert(loc=len(df.columns),column="MahalaDist",value=mahalanobis_distances) 
    return df

def VisualiseMahalanobis(data,exclude): # input: dataframe with mahalanobis distance calculated
    # exclude : excluded feature name (list,str)
    df = data.drop(exclude,axis=1)
    # reduce data dimension to 2 components using PCA
    pca = PCA(n_components=2)
    data_scaled = StandardScaler().fit_transform(df)
    data_reduced = pca.fit_transform(data_scaled)
    df.insert(loc=len(df.columns),column="PC1",value=data_reduced.T[0])
    df.insert(loc=len(df.columns),column="PC2",value=data_reduced.T[1])
    # reinsert excluded columns (features)
    for feature in exclude:
        df.insert(loc=len(df.columns),column=feature,value=data[feature]) 
    # plot outliers
    outliers_mask = df["MahalaDist"] > chi_square
    x = df["PC1"][outliers_mask]
    y = df["PC2"][outliers_mask]
    plt.figure(figsize=(6,6))
    plt.scatter(x,y,label="outliers",color="red")
    # plot non-outliers
    x = df["PC1"][~outliers_mask]
    y = df["PC2"][~outliers_mask]
    plt.scatter(x,y,label="non-outliers",alpha=0.5)
    plt.legend()
    plt.title("PCA-Mahalanobis")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    # save plot
    plt.savefig(f"{output_dir}/multivariate_mahalanobis.pdf")
    # save dataframe containing outliers
    df = df.sort_values(["PC1","PC2"],ascending=[False,False])
    df = df[outliers_mask]
    df.to_csv(f"{output_dir}/multivariate_outliers.csv")
    
# ALGORITHM
if __name__ == '__main__':
    # make output directory
    os.makedirs(output_dir,exist_ok=True)
    # read xlsx, store in dataframe
    df = pd.read_excel(data_raw,sheet_name=sheet_name)
    # print data type per column to check homogeneity
    print(df.dtypes)
    # visualise univariate distribution of each feature to examine the distribution and outliers
    VisualiseUnivariateDist(df,exclude=["Class"])
    # calculate mahalanobis distances
    df_mahalanobis = CalculateMahalanobis(df,exclude=["Class"])
    # visualise multivariate outliers using PCA and mahalanobis distance
    plt.clf() # clear figure
    VisualiseMahalanobis(df_mahalanobis,exclude=["Class","MahalaDist"])