import feedbackModel as fm
import feedbackModelWithSplittedData as fm_splitted_data
from pandas import read_csv
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # read the .data file into a Pandas DataFrame, specifying the delimiter
    # df = read_csv('zoo.data', delimiter=',')
    df = read_csv('nursery.data', delimiter=',')

    # extract input features (X) and target variable (y)
    X = df.iloc[:, :-1].values  # extract all columns except the last one as input features (X)
    y = df.iloc[:, -1].values   # extract the last column as the target variable (y)


    for i in range(1, 4):
        plt.figure()  # create a figure for this teacher
        print(f"\nTeacher{i}:\n")
        for j in range(1, 6):
            print(f"run{j}")
            f_name = f"Teacher{i}_output{j}.txt"
            png_name = f"Teacher{i}_mistakes{j}.png"
            model = fm.feedbackModel(output_file_name=f_name, png_file_name=png_name)
            model.fit(X, y, i)

