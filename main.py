# TODO features_labels_dict!!!!!!!!!!!!!!

import feedbackModel as fm
from pandas import read_csv

if __name__ == "__main__":
    # Read the .data file into a Pandas DataFrame, specifying the delimiter
    df = read_csv('zoo.data', delimiter=',')
    # df = read_csv('nursery.data', delimiter=',')

    # Extract input features (X) and target variable (y)
    X = df.iloc[:, :-1].values  # Extract all columns except the last one as input features (X)
    y = df.iloc[:, -1].values   # Extract the last column as the target variable (y)

    model = fm.feedbackModel()
    model.fit(X, y)
    
    accuracy = model.test()
    
    print("Accuracy:", accuracy)
