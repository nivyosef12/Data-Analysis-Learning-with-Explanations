import numpy as np
from Teacher import Teacher
# from TeacherWithSplittedData import Teacher


class Teacher2(Teacher):
    """
        @:param example is the example the algorithm gets (on each iteration) during the learning phase
                IMPORTANT -> example is already preprocessed!!!
        @:param explanation is the explanation of the algorithm to its predication
        @:param prediction is the algorithm prediction for the example

        @:returns a discriminative_feature in case the predication != to the true label
    """
    def teach(self, example, explanation, prediction):
        try:
            true_label = self.features_labels_dict[tuple(example)]
        except:
            # the example isn't in the dataset
            return None, None
            
        if true_label == prediction:
            return true_label, None

        # print(f"\ntrue label: {true_label}\nprediction: {prediction}")
        # print(f"example: {example}\nexplanation: {explanation}\nexample != explanation: {example != explanation}\n")

        chosen_discriminative_feature = self.mostDiscriminativeFeature(example, explanation, prediction, true_label)
        return true_label, chosen_discriminative_feature  


    """
        @:param example is the example the algorithm gets (on each iteration) during the learning phase
                IMPORTANT -> example is already preprocessed!!!
        @:param explanation is the explanation of the algorithm to its predication
        @:param prediction is the algorithm prediction for the example
        @:param true_label is the true label of the example

        @:returns a discriminative_feature in case the predication != to the true label
    """
    def mostDiscriminativeFeature(self, example, explanation, prediction, true_label):
        different_indexes = np.where(example != explanation)[0]
        print(f"--------------------------------------\ndifferent indexes: {different_indexes}\n")
        max_difference = 0
        chosen_discriminative_feature = None
        for i in different_indexes:
            difference = self.discriminativeFeatureScore(example, prediction, true_label, i)
            if difference >= max_difference:
                print(f"difference for index {i} is {difference}\n")
                max_difference = difference
                chosen_discriminative_feature = [i, example[i]]
        print(f"max difference = {max_difference}, chosen_discriminative_feature = {chosen_discriminative_feature}\n")
        
        if max_difference != 0 and chosen_discriminative_feature is None:
            print("\nproblem!!\n")
            exit(0)
        
        return chosen_discriminative_feature


    """
        @:param example is the example the algorithm gets (on each iteration) during the learning phase
                IMPORTANT -> example is already preprocessed!!!
        @:param prediction is the algorithm prediction for the example
        @:param true_label is the true label of the example
        @:param i is the feature index that we wish to calculate it's score

        @:returns a discriminative_feature in case the predication != to the true label
    """
    def discriminativeFeatureScore(self, example, prediction, true_label, i):
        num_of_stsfyng_exmpls_with_prediction = len(
            {k: v for k, v in self.features_labels_dict.items() if k[i] == example[i] and v == prediction})
        num_of_stsfyng_exmpls_with_true_label = len(
            {k: v for k, v in self.features_labels_dict.items() if k[i] == example[i] and v == true_label})

        prediction_percentage = num_of_stsfyng_exmpls_with_prediction / len(self.features_labels_dict)
        true_label_percentage = num_of_stsfyng_exmpls_with_true_label / len(self.features_labels_dict)
        difference = np.abs(prediction_percentage - true_label_percentage)
        
        return difference