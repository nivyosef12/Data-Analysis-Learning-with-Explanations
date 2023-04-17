# TODO
# 1. what the hell is C[X] !?!?!?!?!

import numpy as np
import Law
import Teacher1 as teacher_1
import Teacher2 as teacher_2


class feedbackModel:
    """
        @:param X is a set of n vectors, each vector is a d-dimensional vector containing 0 or 1
                representing the features
        @:param y is an n-dimensional vector containing the correct labels
        @:param teacher_type is an integer representing the type of teacher
        @:param default_explanation TODO: is it necessary?
        @:param default_label TODO: is it necessary?     
    """

    def __init__(self):

        self.default_explanation = None
        self.default_label = None
        self.laws = []  # list of laws TODO: maybe use dictionary

        # teacher.preprocess(self.default_explanation)

    def fit(self, X, y, teacher_type=1):

        teacher_types = {1: teacher_1.Teacher1, 2: teacher_2.Teacher2}
        if teacher_type not in teacher_types:
            raise ValueError("Invalid teacher_type value")
        teacher = teacher_types[teacher_type](X, y)

        X_legal = teacher.get_X()

        self.default_explanation = X_legal[0]
        self.default_label = y[0]

        for features in X_legal:

            # get prediction and explanation for current example
            prediction, explanation, law = self.__predict(features)

            # get real label and discriminative feature from teacher
            true_label, discriminative_feature = teacher.teach(features, explanation, prediction)

            print(f"predicted: {prediction}, with the explanation of {explanation}\n"
                  f"the teacher response is: {true_label} with {discriminative_feature} as discriminative feature")

            # in case the algorithm to the prediction wrong
            if prediction != true_label:

                if law is None:
                    # update the laws with the new law
                    new_law = Law(explanation, true_label, discriminative_feature)
                    self.laws.append(new_law)

                else:
                    # update the law
                    not_discriminative_feature = (discriminative_feature[0], 1 - discriminative_feature[1])
                    law.updateFeatures(not_discriminative_feature)

    def __predict(self, features):

        for law in self.laws:
            if law.isFitting(features):
                prediction = law.getLabel()
                explanation = law.getExplanation()

                return prediction, explanation, law

        return self.default_label, self.default_explanation, None

    def predict(self, X):
        if self.default_explanation is None:
            raise ValueError("The model hasn't been fitted yet")

        prediction = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            for law in self.laws:
                prediction = law.getLabel() if law.isFitting(X[i]) else self.default_label

        return prediction
