import numpy as np


class Law:
    """
        explanation is a d dimensional vector containing 0 or 1
        label is a label in some set of possible labels
    """

    def __init__(self, explanation, label):
        self.explanation = explanation
        self.label = label

    def equal(self, explanation):
        return np.array_equal(self.explanation, explanation)

    def getExplanation(self):
        return self.explanation

    def getLabel(self):
        return self.label
