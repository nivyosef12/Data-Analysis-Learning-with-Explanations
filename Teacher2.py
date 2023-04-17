import numpy as np
from Teacher import Teacher


class Teacher2(Teacher):
    """

        @:param example is the example the algorithm gets (on each iteration) during the learning phase
                IMPORTANT -> example is already preprocessed!!!
        @:param explanation is the explanation of the algorithm to its predication
        @:param prediction is the algorithm prediction for the example
        
        @:returns a discriminative_feature in case the predication != to the true label
    """

    def teach(self, example, explanation, prediction):
        pass
