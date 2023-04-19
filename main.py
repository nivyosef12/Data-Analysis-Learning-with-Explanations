# TODO features_labels_dict!!!!!!!!!!!!!!
# TODO when testing and the example doesnt satisfy any law, predict the label of the law that is closest

import feedbackModel as fm
import feedbackModelWithSplittedData as fm_splitted_data
from pandas import read_csv

if __name__ == "__main__":
    # read the .data file into a Pandas DataFrame, specifying the delimiter
    # df = read_csv('zoo.data', delimiter=',')
    df = read_csv('nursery.data', delimiter=',')

    # extract input features (X) and target variable (y)
    X = df.iloc[:, :-1].values  # extract all columns except the last one as input features (X)
    y = df.iloc[:, -1].values   # extract the last column as the target variable (y)

    without_splitting_data = True
    # # uncomment for splitted data run
    # # uncomment THIS line: # from TeacherWithSplittedData import Teacher from Teacher1, Teacher2
    # without_splitting_data = False

    if without_splitting_data:
        model = fm.feedbackModel()
        model.fit(X, y)

    else:
        model = fm_splitted_data.feedbackModel()
        model.fit(X, y)

        accuracy = model.test()
        print("Accuracy:", accuracy)
