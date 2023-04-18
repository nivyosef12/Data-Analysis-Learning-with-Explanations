import json

import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Teacher(ABC):
    max_num_of_feature_categories = 100
    num_of_buckets_for_continuous_features = 5

    """
        @:param X are a d dimensional vector containing 0 or 1
        @:param labels are a label in some set of possible labels corresponding with X
                        i.e. feature[i] are labeled a label[i]
    """

    def __init__(self, X, labels):
        X_legal = self.preprocess(X)

        # self.labels = labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_legal, labels)

        self.features_labels_dict = {}
        for features, label in zip(self.X_train, self.y_train):
            if tuple(features) in self.features_labels_dict:
                print("features override in features_labels_dict")
                exit(1)
            self.features_labels_dict[tuple(features)] = label

        # # Open a file for writing
        # with open("xandy_train.txt", "w") as file:
        #     # Iterate over the arrays and write each row to the file
        #     for i in range(self.X_train.shape[0]):
        #         X_train_str = str(list(self.X_train[i]))
        #         y_train_str = str(self.y_train[i])
        #         row_str = X_train_str + ":" + y_train_str
        #         file.write(row_str + "\n")
        #
        # # Open a file for writing
        # with open("features_labal_dict.txt", "w") as file:
        #     # Iterate over the keys and values in the dictionary and write each key-value pair to the file
        #     for key, value in self.features_labels_dict.items():
        #         row_str = str(key) + ":" + str(value)
        #         file.write(row_str + "\n")

    """
        Convert non-binary attributes in X to binary attributes.

        @param X: array-like, shape (n_samples, n_features)
            Input data.

        @return: binary_X: array, shape (n_samples, n_binary_features)
            Converted binary data.
    """

    def preprocess(self, X):
        binary_X = np.empty((X.shape[0], 0))

        for i in range(X.shape[1]):
            col = X[:, i]
            # Handle missing values
            # TODO handle missing values
            # col[np.isnan(col)] = 0

            if col.dtype == np.bool_:
                # Boolean attribute
                col_binary = col.astype(int)

            else:
                if col.dtype == object:
                    # Categorical attribute with string values
                    le = LabelEncoder()
                    col = le.fit_transform(col)
                    # col_binary = np.eye(len(np.unique(col_numeric)))[col_numeric]

                # TODO: explain this with comments
                if len(np.unique(col)) <= self.max_num_of_feature_categories:
                    # Categorical attribute with few categories
                    unique_values = np.unique(col)
                    num_unique_values = len(unique_values)
                    col_binary = np.zeros((len(col), num_unique_values), dtype=int)
                    for j, unique_value in enumerate(unique_values):
                        col_binary[col == unique_value, j] = 1

                else:
                    # Continuous attribute with many possible values
                    bucket_size = 100 / self.num_of_buckets_for_continuous_features
                    buckets = np.array(range(self.num_of_buckets_for_continuous_features)) * bucket_size

                    col_binary = np.empty((len(col), self.num_of_buckets_for_continuous_features))
                    for j in range(buckets.size):
                        if j < self.num_of_buckets_for_continuous_features - 1:
                            col_binary[:, j] = np.where(
                                (col > np.percentile(col, buckets[j])) & (col <= np.percentile(col, buckets[j + 1])),
                                1, 0)
                        else:
                            col_binary[:, j] = np.where(col > np.percentile(col, buckets[j]), 1, 0)

            binary_X = np.column_stack((binary_X, col_binary))

        return binary_X

    def get_preprocessed_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test



    """

        @:param example is the example the algorithm gets (on each iteration) during the learning phase
                IMPORTANT -> example is already preprocessed!!!
        @:param explanation is the explanation of the algorithm to its predication
        @:param prediction is the algorithm prediction for the example
        
        @:returns a discriminative_feature in case the predication != to the true label
    """

    @abstractmethod
    def teach(self, example, explanation, prediction):
        pass

    def getLabel(self, k):
        return self.features_labels_dict[k]