import numpy as np


class Law:
    """
        @:param explanation is a d dimensional vector containing 0 or 1
        @:param label is a label in some set of possible labels
        @:param feature is a tuple (i, v) where i is in range of 1 to d, and v is {1, -1}
                            now its {1, 0} -> TODO change?
    """

    def __init__(self, explanation, label, feature):
        self.explanation = explanation  # TODO needs to be updated?
        self.label = label
        self.features = np.array(feature)

    """
    @:param new_sample is a d dimensional vector containing 0 or 1
                IMPORTANT -> new_sample is already preprocessed!!!
    @:return true if for each (i, v) in self.features new_sample[i] == v
    """
    def isFitting(self, new_sample):
        # create a boolean mask to check if the condition holds true for each (i, v) tuple
        mask = [new_sample[i] == v for i, v in self.features]

        # check if all the values in the boolean mask are True
        return np.all(mask)

    def updateFeatures(self, discriminative_feature):
        self.features = np.append(self.features, discriminative_feature)

    def getExplanation(self):
        return self.explanation

    def getLabel(self):
        return self.label

    def getFeatures(self):
        return self.features
