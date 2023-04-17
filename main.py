import feedbackModel
from pandas import read_csv

if __name__ == __main__:
    # Read the .data file into a Pandas DataFrame, specifying the delimiter
    df = pd.read_csv('zoo.data', delimiter=',')

    # Extract input features (X) and target variable (y)
    X = df.iloc[:, :-1].values  # Extract all columns except the last one as input features (X)
    y = df.iloc[:, -1].values   # Extract the last column as the target variable (y)

    default_explanation = []  # TODO: what is it supposed to be and is it supposed to be here?
    default_label = 0  # TODO: what is it supposed to be and is it supposed to be here?
    
    model = feedbackModel(default_explanation, default_label)
    model.fit(X, y)