# TODO
# 1. what the hell is C[X] !?!?!?!?!
# 2. why the fuck is the teacher returns the true label ?!?!?!?!?
# 3. handle code duplication

import numpy as np

from Law import Law


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

        for features, label in np.ndier([X, y]):
            # found = False

            for law in self.laws:
                if law.isFitting(features):
                    # found = True
                    predication = law.getLabel()
                    explanation = law.getExplanation()

                    print(f"predicted: {predication}, with the explanation of {explanation}")

                    if predication != label:
                        # get discriminative feature from teacher
                        discriminative_feature = self.teacher(features, label, explanation, predication)

                        # update the law
                        law.updateFeatures(discriminative_feature)
                        # TODO anything else??? , should break???
                        break

            # if not found:
            # this block of code is executed if the inner loop completes without hitting a break statement
            else:
                predication = self.default_label
                explanation = self.default_explanation

                print(f"predicted: {predication}, with the explanation of {explanation}")

                if predication != label:
                    # get discriminative feature from teacher
                    discriminative_feature = self.teacher(features, label, explanation, predication)

                    # update the laws with the new law
                    new_law = Law(explanation, label, discriminative_feature)
                    self.laws.append(new_law)
                    pass


"""

for i in range(10):
    for j in range(5):
        if i < 5 or i > 7:
            if i < 5:
                print("break")
                break
        print(f"{i}, {j}")
    else:
        # this block of code is executed if the inner loop completes without hitting a break statement
        print("inner loop completed")

    print("------------------")



"""
