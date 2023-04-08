# TODO
# 1. what the hell is C[X] !?!?!?!?!

import numpy as np

class feedbackModel:

    def __init__(self, teacher, default_explanation, default_label):
        self.teacher = teacher
        self.default_explanation = default_explanation
        self.default_label = default_label
        self.laws = []  # list of laws TODO: maybe use dictionary

    """
        X is a set of vectors, each vector is a d dimensional vector containing 0 or 1
                                    representing the features
                                    
        y is a vector of labels, each label y_i correspond with a vector of features x_i in X
    """
    def learn(self, X, y):

        for feature, label in np.ndier([X, y]):
            found = False

            for law in self.laws:
                if law.equal(feature):
                    found = True
                    predication = law.getLabel()
                    explanation = law.getExplanation()

                    print(f"predicted: {predication}, with the explanation of {explanation}")

                    if predication != label:
                        # get discriminative feature from teacher
                        # update the law
                        # anything else ???
                        pass

            if not found:
                predication = self.default_label
                explanation = self.default_explanation

                print(f"predicted: {predication}, with the explanation of {explanation}")

                if predication != label:
                    # get discriminative feature from teacher
                    # new_law = Law(discriminative feature, label)
                    # self.L.add(new_law)
                    # anything else ???
                    pass


