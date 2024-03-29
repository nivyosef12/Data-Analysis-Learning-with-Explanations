import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


class Teacher(ABC):
    """
        @:param X are a d dimensional vector containing 0 or 1
        @:param labels are a label in some set of possible labels corresponding with X
                        i.e. feature[i] are labeled a label[i]
        @:param max_num_of_feature_categories is the max number of categories that we consider as discrete
        @:param num_of_buckets_for_continuous_features is the number of buckets we create for continuous value categories
    """

    def __init__(self, X, labels, max_num_of_feature_categories=20, num_of_buckets_for_continuous_features=10):
        self.max_num_of_feature_categories = max_num_of_feature_categories
        self.num_of_buckets_for_continuous_features = num_of_buckets_for_continuous_features

        self.X = self.preprocess(X)

        self.features_labels_dict = {}
        for features, label in zip(self.X, labels):

            # by putting continuous values in buckets, we might end up with 2 identical feature vectors with
            # different labels
            if tuple(features) in self.features_labels_dict and label != self.features_labels_dict[tuple(features)]:
                print("features override in features_labels_dict:\n")
                print(
                    f"the example\n{features}\nappears twice with different labels {label}, {self.features_labels_dict[tuple(features)]}\n")
                exit(1)
            self.features_labels_dict[tuple(features)] = label

    """
        Convert non-binary attributes in X to binary attributes.

        @param X: array-like, shape (n_samples, n_features)
            Input data.

        @return: binary_X: array, shape (n_samples, n_binary_features)
            Converted binary data.
    """

    def preprocess(self, X):

        # impute missing values
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit_transform(X)

        binary_X = np.empty((X.shape[0], 0))

        for i in range(X.shape[1]):
            col = X[:, i]

            if np.array_equal(col, col.astype(bool)):
                # Binary attribute
                col_binary = col

            elif col.dtype == np.bool_:
                # Boolean attribute
                col_binary = col.astype(int)

            else:
                # categorical attribute with string values
                if col.dtype == object:
                    le = LabelEncoder()
                    col = le.fit_transform(col)
                    # col_binary = np.eye(len(np.unique(col_numeric)))[col_numeric]

                unique_values = np.unique(col)  # the unique values in this column

                # categorical attribute with few categories
                if len(unique_values) <= self.max_num_of_feature_categories:
                    col_binary = np.zeros((len(col), len(unique_values)),
                                          dtype=int)  # create a zero matrix with a column for every unique value
                    for j, unique_value in enumerate(unique_values):
                        # write 1 in the column that corresponds to the enumeration of the unique value
                        col_binary[col == unique_value, j] = 1

                # continuous attribute with many possible values
                else:
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
        return self.X

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
        return self.features_labels_dict[tuple(k)]
