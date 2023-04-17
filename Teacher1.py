import numpy as np
from abc import ABC, abstractmethod
import Teacher

class Teacher1(Teacher):
        
    """

        @:param example is the example the algorithm gets (on each iteration) during the learning phase
                IMPORTANT -> example is already preprocessed!!!
        @:param explanation is the explanation of the algorithm to its predication
        @:param prediction is the algorithm prediction for the example
        
        @:returns a discriminative_feature in case the predication != to the true label
    """
    def teach(self, example, explanation, prediction):
        true_label = self.features_labels_dict[example]
        if true_label == prediction:
            return true_label, None

        different_indexes = np.where(example != explanation)[0]
        chosen_index = np.random.choice(different_indexes)
        return true_label, (chosen_index, example[chosen_index])  # TODO return tuple?
