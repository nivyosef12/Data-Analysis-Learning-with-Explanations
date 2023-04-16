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

    def learn(self, X):

        for features in X:

            # TODO check if feature changes outside the preprocess function
            self.teacher.preprocess(features)

            # get prediction and explanation for current example
            predication, explanation, law = self.predict(features)

            # get real label and discriminative feature from teacher
            true_label, discriminative_feature = self.teacher.teach(features, explanation, predication)

            print(f"predicted: {predication}, with the explanation of {explanation}\n"
                  f"the teacher response is: {true_label} with {discriminative_feature} as discriminative feature")

            # in case the algorithm to the prediction wrong
            if predication != true_label:

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

                return predication, explanation, law

        return self.default_label, self.default_explanation, None
