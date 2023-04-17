import feedbackModel
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # Read the .data file into a Pandas DataFrame, specifying the delimiter
    df = pd.read_csv('zoo.data', delimiter=',')

    # Extract input features (X) and target variable (y)
    X = df.iloc[:, :-1].values  # Extract all columns except the last one as input features (X)
    y = df.iloc[:, -1].values   # Extract the last column as the target variable (y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = feedbackModel()
    model.fit(X_train, y_train)
    
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    
    print("Accuracy:", accuracy)
