import numpy as np
from Law import Law
from Teacher1 import Teacher1
from Teacher2 import Teacher2
from Teacher3 import Teacher3
from Teacher4 import Teacher4
from sklearn.metrics import accuracy_score
# from sklearn.utils import shuffle
from random import randint
from tqdm import tqdm

class feedbackModel:

    def __init__(self, output_file_name="output.txt", optimized=False):

        self.default_explanation = None
        self.default_label = None
        self.laws = []  # list of laws 
        self.teacher = None
        self.num_of_mistakes = 0
        self.mistakes_made = []
        self.output_file_name = output_file_name
        self.f = None
        self.optimized = optimized
        
    def fit(self, X, y, teacher_type=1):
        # open output file
        self.f = open(self.output_file_name, "a")
        self.f.truncate(0)  # erase the contents of the file if it already existed
        
        # shuffle the data
        indexes = np.arange(y.shape[0])
        np.random.shuffle(indexes)
        shuffled_X = X[indexes]
        shuffled_y = y[indexes]

        teacher_types = {1: Teacher1, 2: Teacher2, 3:Teacher3, 4:Teacher4}
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
        
        # initialize prediction list for debugging purposes
        prediction_list = []

        self.f.write("-------------------- training --------------------\n\n")
        for features in tqdm(preprocessed_data):
            
            # predict
            if self.optimized:
                prediction, explanation, law = self.__predict2(features)
            else:
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
                    new_law = Law(features, true_label, discriminative_feature)
                    self.laws.append(new_law)

                else:
                    # update the law
                    discriminative_feature[:, 1] = 1 - discriminative_feature[:, 1]
                    law.updateFeatures(discriminative_feature)

            # for generating the graph
            self.mistakes_made.append(self.num_of_mistakes)
            prediction_list.append(prediction)

        examples_seen, percent_mistakes = self.plotAndPrint(prediction_list, y)
        
        # close the output file
        self.f.close()
        
        return examples_seen, percent_mistakes
        
        
    def plotAndPrint(self, prediction_list, y):
        self.f.write("\n-------------------- printing the final decision list --------------------\n")
        for p, l in zip(prediction_list, y):
            answer = "RIGHT" if p == l else "WRONG"
            self.f.write(f"prediction: {p},\ttrue label: {l}\t--> {answer}\n")

        num_of_example = y.shape[0]
        self.f.write(f"\npercentage of mistake on the entire data set: {self.num_of_mistakes} / {num_of_example} ="
              f" {(self.num_of_mistakes / num_of_example) * 100}%")

        examples_seen = [i + 1 for i in range(num_of_example)]

        # calculate the percentage of mistakes made
        percent_mistakes = [(m / e) * 100 for m, e in zip(self.mistakes_made, examples_seen)]

        return examples_seen, percent_mistakes


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
        
        return closest_pred, closest_expl, None