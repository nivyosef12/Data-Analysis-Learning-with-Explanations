import numpy as np
import matplotlib.pyplot as plt
from Law import Law
from Teacher1 import Teacher1
from Teacher2 import Teacher2
from Teacher3 import Teacher3
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from random import randint

class feedbackModel:

    def __init__(self, output_file_name="output.txt", png_file_name="mistakes_over_time.png"):

        self.default_explanation = None
        self.default_label = None
        self.laws = []  # list of laws 
        self.teacher = None
        self.num_of_mistakes = 0
        self.mistakes_made = []
        self.output_file_name = output_file_name
        self.png_file_name = png_file_name
        self.f = None
        
    def fit(self, X, y, teacher_type=1):
        # open output file
        self.f = open(self.output_file_name, "a")
        self.f.truncate(0)  # erase the contents of the file if it already existed
        
        # shuffle the data
        shuffled_X, shuffled_y = shuffle(X, y)

        teacher_types = {1: Teacher1, 2: Teacher2, 3:Teacher3}
        if teacher_type not in teacher_types:
            raise ValueError("Invalid teacher_type value")

        # initialize teacher
        self.teacher = teacher_types[teacher_type](shuffled_X, shuffled_y)

        # get processed data from teacher
        preprocessed_data = self.teacher.get_preprocessed_data()

        # initialize default label and explanation
        i = randint(0, preprocessed_data.shape[0] - 1)  # randomly select an index
        self.default_label = shuffled_y[i]
        self.default_explanation = preprocessed_data[i]
        
        print(f"default_label = {self.default_label}\ndefault explanation =\n{self.default_explanation}\n")

        # initialize prediction list for debugging purposes
        prediction_list = []

        self.f.write("-------------------- training --------------------\n\n")
        for features in preprocessed_data:
            # predict
            prediction, explanation, law = self.__predict(features)

            # get true label and discriminative feature from teacher
            true_label, discriminative_feature = self.teacher.teach(features, explanation, prediction)
            self.f.write("\n-------------------- curr iteration information --------------------\n")
            self.f.write(f"the example: {features}\n"
                  f"predicted: {prediction}, with the explanation of\n{explanation}\n"
                  f"the teacher response is: {true_label} with {discriminative_feature} as discriminative feature\n")

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

        self.plotAndPrint(prediction_list, y)
        
        # close the putput file
        self.f.close()
        
        
    def plotAndPrint(self, prediction_list, y):
        self.f.write("\n-------------------- printing list of laws --------------------\n")
        for law in self.laws:
            self.f.write(f"law-> label = {law.getLabel()}\n"
                  f"      features =\n{law.getFeatures()}\n"
                  f"      explanation =\n{law.getExplanation()}\n")

        for p, l in zip(prediction_list, y):
            answer = "RIGHT" if p == l else "WRONG"
            self.f.write(f"prediction: {p},\ttrue label: {l}\t--> {answer}\n")

        num_of_example = y.shape[0]
        self.f.write(f"\npercentage of mistake on the entire data set: {self.num_of_mistakes} / {num_of_example} ="
              f" {(self.num_of_mistakes / num_of_example) * 100}%")

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
        plt.savefig(self.png_file_name)

    
    def __predict(self, features):
        for law in self.laws:
            if law.isFitting(features):
                prediction = law.getLabel()
                explanation = law.getExplanation()

                return prediction, explanation, law

        return self.default_label, self.default_explanation, None


    # instead of the default label, we predict based on the closest law
    def __predict2(self, features):
        closest_pred = self.default_label
        closest_expl = self.default_explanation
        closest_law = None
        max_num_of_matching_dfs = 0  # max number of discriminative features of a law that matched our features
        for law in self.laws:
            prediction = law.getLabel()
            explanation = law.getExplanation()
            
            if law.isFitting(features):
                return prediction, explanation, law
            
            num_of_matching_dfs = law.numOfMatchingFeatures(features)
            if num_of_matching_dfs > max_num_of_matching_dfs:
                max_num_of_matching_dfs = num_of_matching_dfs
                closest_pred = prediction
                closest_expl = explanation
                closest_law = law
                
        print(f"\n\nexample = {features}\n")
        
        if closest_law:
            print(f"law-> explanation =\n{closest_law.getExplanation()}\n"
                f"      features =\n{closest_law.getFeatures()}\n"
                f"      label = {closest_law.getLabel()}\n")
        
        return closest_pred, closest_expl, None