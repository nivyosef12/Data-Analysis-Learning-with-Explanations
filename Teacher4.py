import numpy as np
from Teacher import Teacher


class Teacher4(Teacher):
    def __init__(self, X, labels):
        super().__init__(X, labels)

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

        chosen_discriminative_feature = self.mostDiscriminativeFeature(example, explanation, prediction, true_label)
        return true_label, np.array(chosen_discriminative_feature, dtype=int)

    """
        @:param example is the example the algorithm gets (on each iteration) during the learning phase
                IMPORTANT -> example is already preprocessed!!!
        @:param explanation is the explanation of the algorithm to its predication
        @:param prediction is the algorithm prediction for the example
        @:param true_label is the true label of the example

        @:returns a random number discriminative_features in case the predication != to the true label
    """

    def mostDiscriminativeFeature(self, example, explanation, prediction, true_label):
        different_indexes = np.where(example != explanation)[0]

        # randomly select a number of discriminative features, with a higher probability of selecting a lower number
        probabilities = np.array([x ** 3 for x in range(-len(different_indexes), 0)], dtype=float)
        probabilities /= np.sum(probabilities)
        num_of_dfs = np.random.choice(list(range(1, len(different_indexes) + 1)), p=probabilities)

        chosen_dfs = np.empty((0,), dtype=int)
        differences = np.empty((0,))
        min_difference = 0
        for i in different_indexes:

            difference = self.discriminativeFeatureScore(example, prediction, true_label, i)

            # if we haven't chosen enough features yet
            if chosen_dfs.shape[0] < num_of_dfs:
                chosen_dfs = np.append(chosen_dfs, [i])
                differences = np.append(differences, difference)

            # if this feature is better than the worst feature in chosen_dfs
            elif difference > min_difference:
                min_index = np.argmin(differences)
                differences[min_index] = difference
                chosen_dfs[min_index] = i
                min_difference = np.min(differences)

        output = [[i, example[i]] for i in chosen_dfs.tolist()]
        return output

    """
        @:param example is the example the algorithm gets (on each iteration) during the learning phase
                IMPORTANT -> example is already preprocessed!!!
        @:param prediction is the algorithm prediction for the example
        @:param true_label is the true label of the example
        @:param i is the feature index that we wish to calculate it's score

        @:returns a discriminative_feature in case the predication != to the true label
    """

    def discriminativeFeatureScore(self, example, prediction, true_label, i):

        num_of_satisfying_examples_with_prediction = len(
            [1 for k, v in self.features_labels_dict.items() if k[i] == example[i] and v == prediction])

        num_of_satisfying_examples_with_true_label = len(
            [1 for k, v in self.features_labels_dict.items() if k[i] == example[i] and v == true_label])

        prediction_percentage = num_of_satisfying_examples_with_prediction / len(self.features_labels_dict)
        true_label_percentage = num_of_satisfying_examples_with_true_label / len(self.features_labels_dict)
        difference = np.abs(prediction_percentage - true_label_percentage)

        return difference
