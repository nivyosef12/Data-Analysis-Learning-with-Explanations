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
        true_label = self.features_labels_dict[tuple(example)]
        if true_label == prediction:
            return true_label, None

        # print(f"\ntrue label: {true_label}\nprediction: {prediction}")
        # print(f"example: {example}\nexplanation: {explanation}\nexample != explanation: {example != explanation}\n")

        chosen_discriminative_feature = self.most_discriminative_feature(example, explanation, prediction, true_label)
        return true_label, chosen_discriminative_feature  # TODO return tuple?

    def most_discriminative_feature(self, example, explanation, prediction, true_label):
        different_indexes = np.where(example != explanation)[0]
        max_difference = 0
        chosen_discriminative_feature = None
        for i in different_indexes:
            num_of_stsfyng_exmpls_with_prediction = len(
                {k: v for k, v in self.features_labels_dict.items() if k[i] == example[i] and v == prediction})
            num_of_stsfyng_exmpls_with_true_label = len(
                {k: v for k, v in self.features_labels_dict.items() if k[i] == example[i] and v == true_label})

            prediction_percentage = num_of_stsfyng_exmpls_with_prediction / len(self.features_labels_dict)
            true_label_percentage = num_of_stsfyng_exmpls_with_true_label / len(self.features_labels_dict)
            difference = np.abs(prediction_percentage - true_label_percentage)

            if difference > max_difference:
                max_difference = difference
                chosen_discriminative_feature = [i, example[i]]

        return chosen_discriminative_feature