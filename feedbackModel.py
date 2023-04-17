# TODO
# 1. what the hell is C[X] !?!?!?!?!

import numpy as np
import Law
import Teacher1
import Teacher2


class feedbackModel:

    """
        @:param X is a set of n vectors, each vector is a d-dimensional vector containing 0 or 1
                representing the features
        @:param y is an n-dimensional vector containing the correct labels
        @:param teacher_type is an integer representing the type of teacher
        @:param default_explanation TODO: is it necessary?
        @:param default_label TODO: is it necessary?     
    """
    def __init__(self, default_explanation, default_label):
        
        self.default_explanation = default_explanation
        self.default_label = default_label
        self.laws = []  # list of laws TODO: maybe use dictionary

        # teacher.preprocess(self.default_explanation)
    

    def fit(self, X, y, teacher_type=1):
        
        teacher_types = {1:Teacher1, 2:Teacher2}
        if teacher_type not in teacher_types:
            raise ValueError("Invalid teacher_type value")
        teacher = teacher_types[teacher_type](X, y)
        
        X = teacher.get_X()
        
        for features in X:

            # get prediction and explanation for current example
            predication, explanation, law = self.predict(features)

            # get real label and discriminative feature from teacher
            true_label, discriminative_feature = teacher.teach(features, explanation, predication)

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
