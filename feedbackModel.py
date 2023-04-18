# TODO
# 1. what the hell is C[X] !?!?!?!?!

import numpy as np
from Law import Law
import Teacher1 as teacher_1
import Teacher2 as teacher_2
from sklearn.metrics import accuracy_score


class feedbackModel:
    """

    """

    def __init__(self):

        self.default_explanation = None
        self.default_label = None
        self.laws = []  # list of laws TODO: maybe use dictionary
        self.teacher = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # teacher.preprocess(self.default_explanation)

    def fit(self, X, y, teacher_type=1):

        teacher_types = {1: teacher_1.Teacher1, 2: teacher_2.Teacher2}
        if teacher_type not in teacher_types:
            raise ValueError("Invalid teacher_type value")
        self.teacher = teacher_types[teacher_type](X, y)

        self.X_train, self.X_test, self.y_train, self.y_test = self.teacher.get_preprocessed_data()

        zero_indices = np.where(self.y_train == 3)[0]  # Get the indices where y is 0
        random_index = np.random.choice(zero_indices)  # Choose a random index from the zero indices

        # Access the corresponding values of x and y using the random_index
        self.default_explanation = self.X_train[random_index]
        self.default_label = self.y_train[random_index]

        # self.default_explanation = self.X_train[0]
        # self.default_label = self.y_train[0]
        print(self.default_label)

        # np.random.shuffle(self.X_train)
        for features in self.X_train:

            # get prediction and explanation for current example
            # print("xt")
            # print(features)
            prediction, explanation, law = self.__predict(features)

            # get real label and discriminative feature from teacher
            true_label, discriminative_feature = self.teacher.teach(features, explanation, prediction)
            # print("--------------------------------")
            # print(f"predicted: {prediction}, with the explanation of {explanation}\n"
            #       f"the teacher response is: {true_label} with {discriminative_feature} as discriminative feature")
            # print(f"law is {law}")

            # in case the algorithm to the prediction wrong
            if prediction != true_label:
                # print("explanation is")
                # print(explanation)
                if law is None:
                    # update the laws with the new law
                    new_law = Law(explanation, true_label, discriminative_feature)
                    # print(f"-----\ncreating law with list of discriminative_feature is {new_law.getFeatures()}")
                    self.laws.append(new_law)

                else:
                    # update the law
                    not_discriminative_feature = np.array([discriminative_feature[0], 1 - discriminative_feature[1]])
                    # print(f"----\nnot discriminative_feature is {discriminative_feature}")
                    law.updateFeatures(not_discriminative_feature)
                    # print(f"list of discriminative_feature is {law.getFeatures()}")

    def __predict(self, features):
        for law in self.laws:
            # print("curr law features")
            # print(law.getFeatures())
            if law.isFitting(features):
                prediction = law.getLabel()
                explanation = law.getExplanation()

                return prediction, explanation, law
        # print("did not find")

        return self.default_label, self.default_explanation, None

    def test(self):
        if self.default_explanation is None:
            raise ValueError("The model hasn't been fitted yet")
        prediction = np.empty(self.X_test.shape[0], dtype=np.array([self.default_label]).dtype)
        for i in range(self.X_test.shape[0]):
            for law in self.laws:
                prediction[i] = law.getLabel() if law.isFitting(self.X_test[i]) else self.default_label

        print(f"prediction: {prediction}")
        # get frequencies
        prediction_frequencies = {d: 0 for d in set(prediction)}

        for i in prediction:
            prediction_frequencies[i] += 1

        print(prediction_frequencies)
        accuracy = accuracy_score(self.y_test, prediction)
        return accuracy

