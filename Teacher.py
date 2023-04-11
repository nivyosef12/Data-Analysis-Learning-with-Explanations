import numpy as np


class Teacher:
    """
        @:param features are a d dimensional vector containing 0 or 1
        @:param labels are a label in some set of possible labels corresponding with features
                        i.e. feature[i] are labeled a label[i]
    """

    def __int__(self, features, labels):
        self.features = features
        self.labels = labels

        for feature in self.features:
            self.preprocess(feature)

    def preprocess(self, feature):  # TODO check if feature changes outside the function
        pass

    """
        @:param example is the example the algorithm gets (on each iteration) during the learning phase
                IMPORTANT -> example is already preprocessed!!!
        @:param explanation is the explanation of the algorithm to its predication
        @:param prediction is the algorithm prediction for the example
        
        @:returns a discriminative_feature in case the predication != to the true label
    """
    def teach(self, example, explanation, prediction):
        pass
