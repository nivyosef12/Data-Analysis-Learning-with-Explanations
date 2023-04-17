import numpy as np
from Teacher import Teacher


class Teacher1(Teacher):
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
        print(example)
        true_label = self.features_labels_dict[tuple(example)]
        if true_label == prediction:
            return true_label, None

        print(f"\ntrue label: {true_label}\nprediction: {prediction}")
        print(f"example: {example}\nexplanation: {explanation}\nexample != explanation: {example != explanation}\n")

        different_indexes = np.where(example != explanation)[0]
        chosen_index = np.random.choice(different_indexes)
        return true_label, [chosen_index, example[chosen_index]]  # TODO return tuple?


