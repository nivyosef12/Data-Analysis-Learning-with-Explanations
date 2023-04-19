# TODO
# 1. what the hell is C[X] !?!?!?!?!

import numpy as np
import matplotlib.pyplot as plt
from Law import Law
import Teacher1 as teacher_1
import Teacher2 as teacher_2
from sklearn.metrics import accuracy_score


class feedbackModel:

    def __init__(self):

        self.default_explanation = None
        self.default_label = None
        self.laws = []  # list of laws TODO: maybe use dictionary
        self.teacher = None
        self.num_of_mistakes = 0
        self.mistakes_made = []

    def fit(self, X, y, teacher_type=1):

        teacher_types = {1: teacher_1.Teacher1, 2: teacher_2.Teacher2}
        if teacher_type not in teacher_types:
            raise ValueError("Invalid teacher_type value")

        # initialize teacher
        self.teacher = teacher_types[teacher_type](X, y)

        # get processed data from teacher
        preprocessed_data = self.teacher.get_preprocessed_data()

        # initialize default label and explanation
        self.default_label = y[0]
        self.default_explanation = preprocessed_data[0]

        # initialize prediction list for debugging purposes
        prediction_list = []

        print("-------------------- training --------------------\n")
        for features in preprocessed_data:
            # predict
            prediction, explanation, law = self.__predict(features)

            # get real label and discriminative feature from teacher
            true_label, discriminative_feature = self.teacher.teach(features, explanation, prediction)
            print("-------------------- curr iteration information --------------------")
            print(f"the example: {features}\n"
                  f"predicted: {prediction}, with the explanation of {explanation}\n"
                  f"the teacher response is: {true_label} with {discriminative_feature} as discriminative feature")

            # in case the algorithm prediction was wrong
            if prediction != true_label:
                self.num_of_mistakes += 1
                if law is None:
                    # create new law
                    new_law = Law(explanation, true_label, discriminative_feature)
                    self.laws.append(new_law)

                else:
                    # update the law
                    not_discriminative_feature = np.array([discriminative_feature[0], 1 - discriminative_feature[1]])
                    law.updateFeatures(not_discriminative_feature)

            # for generating the graph
            self.mistakes_made.append(self.num_of_mistakes)
            prediction_list.append(prediction)

        print("\n-------------------- printing list of laws --------------------\n")
        for law in self.laws:
            print(f"law-> label = {law.getLabel()}\n"
                  f"      features = {law.getFeatures()}\n"
                  f"      explanation =  {law.getExplanation()}\n")
        for p, l in zip(prediction_list, y):
            answer = "RIGHT" if p == l else "WRONG"
            print(f"prediction: {p},\ttrue label: {l}\t--> {answer}")

        num_of_example = y.shape[0]
        print(f"\npercentage of mistake on the entire data set: {self.num_of_mistakes} / {num_of_example} ="
              f" {(self.num_of_mistakes / num_of_example) * 100} %")

        examples_seen = [i + 1 for i in range(num_of_example)]

        # calculate the percentage of mistakes made
        percent_mistakes = [(m / e) * 100 for m, e in zip(self.mistakes_made, examples_seen)]

        # create a line plot
        plt.plot(examples_seen, percent_mistakes)

        # set the title and axis labels
        plt.title("Mistakes made over time")
        plt.xlabel("Examples seen")
        plt.ylabel("Percentage of mistakes")

        # save as .png file
        plt.savefig("mistakes_over_time.png")

    def __predict(self, features):
        for law in self.laws:
            if law.isFitting(features):
                prediction = law.getLabel()
                explanation = law.getExplanation()

                return prediction, explanation, law

        return self.default_label, self.default_explanation, None
