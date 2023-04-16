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

        self.teacher.preprocess(self.default_explanation)
    """
        @:param X is a set of vectors, each vector is a d dimensional vector containing 0 or 1
                representing the features                                    
    """

    def learn(self, X, y):

        for features, label in np.ndier([X, y]):

            # TODO check if feature changes outside the preprocess function
            self.teacher.preprocess(features)

            # get prediction and explanation for current example
            predication, explanation, law = self.predict(features)

            # in case the algorithm to the prediction wrong
            if predication != label:

                # get real label and discriminative feature from teacher
                true_label, discriminative_feature = self.teacher.teach(features, explanation, predication)

                if law is None:
                    # update the laws with the new law
                    new_law = Law(explanation, true_label, discriminative_feature)
                    self.laws.append(new_law)

                else:
                    # update the law
                    not_discriminative_feature = (discriminative_feature[0], 1 - discriminative_feature[1])
                    law.updateFeatures(not_discriminative_feature)

    def predict(self, features):

        for law in self.laws:
            if law.isFitting(features):

                predication = law.getLabel()
                explanation = law.getExplanation()

                print(f"predicted: {predication}, with the explanation of {explanation}")
                return predication, explanation, law

        print(f"DEFAULT: predicted: {self.default_label}, with the explanation of {self.default_explanation}")
        return self.default_label, self.default_explanation, None


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

                print(f"DEFAULT: predicted: {predication}, with the explanation of {explanation}")

                if predication != label:
                    # get discriminative feature from teacher
                    discriminative_feature = self.teacher(features, label, explanation, predication)

                    # update the laws with the new law
                    new_law = Law(explanation, label, discriminative_feature)
                    self.laws.append(new_law)
                    pass
                    
                    
                    ------------------ sec version -----------------
                    
    def learn(self, X, y):

        for features, label in np.ndier([X, y]):
            predication, explanation, law = self.predict(features)

            if predication != label:
                # get discriminative feature from teacher
                discriminative_feature = self.teacher(features, label, explanation, predication)

                if law is None:
                    # update the laws with the new law
                    new_law = Law(explanation, label, discriminative_feature)
                    self.laws.append(new_law)

                else:
                    # update the law
                    law.updateFeatures(discriminative_feature)


"""
