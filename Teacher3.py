import numpy as np
from Teacher import Teacher
from Teacher1 import Teacher1
from Teacher2 import Teacher2
from random import choice, sample, randrange
# from TeacherWithSplittedData import Teacher


class Teacher3(Teacher):
    def __init__(self, X, labels, num_of_teachers=7, sample_percentage=0.8):
        super().__init__(X, labels)
        self.num_of_teachers = num_of_teachers  # number of teachers in the ensemble
        self.sample_percentage = sample_percentage  # percentage of samples to input to each teacher
        self.teachers = self.initialize_teachers()
        

    """
        @:returns  a list of num_of_teachers teachers by randomly sampling (without repetition) sample_percentage of data points
    """
    def initialize_teachers():
        num_of_samples = self.X.shape[0] * self.sample_percentage
        
        # initialize a list where indexes_sampled_for_teachers[i] are the indexes of X, y we will input to teachers[i]
        indexes_sampled_for_teachers = [None for i in range(self.num_of_teachers)]
        
        all_indexes = range(self.X.shape[0])
        not_sampled_indexes = set(range(self.X.shape[0]))
        
        for i in range(num_of_teachers):
            sampled_indexes = sample(all_indexes, num_of_samples)
            not_sampled_indexes -= set(sampled_indexes)
            # not_sampled_indexes = [j for j in not_sampled_indexes if j not in sampled_indexes]
            indexes_sampled_for_teachers[i] = sampled_indexes
            
        # make sure that all indexes are selected for at least one teacher
        for i in not_sampled_indexes:
            selected_teachers = sample(range(self.num_of_teachers), randrange(self.num_of_teachers)) # randomly select a random number of teachers
            for j in selected_teachers: # add the index i to the selected teachers
                indexes_sampled_for_teachers[j].append(i)
                
        teachers = [Teacher2(self.X[indexes_sampled_for_teachers[i]], labels[indexes_sampled_for_teachers[i]])]
        return teachers
    
    
    """
        @:param example is the example the algorithm gets (on each iteration) during the learning phase
                IMPORTANT -> example is already preprocessed!!!
        @:param explanation is the explanation of the algorithm to its predication
        @:param prediction is the algorithm prediction for the example

        @:returns the true label and a discriminative_feature in case the predication != to the true label
    """
    def teach(self, example, explanation, prediction):
        
        true_label = self.features_labels_dict[tuple(example)]
        if true_label == prediction:
            return true_label, None
        
        results = {}
        for teacher in self.teachers:
            label, discriminative_feature = teacher.teach(example, explanation, prediction)
            if(label == None):
                # this teacher doesn't have this example in it's dataset
                continue
            
            # get the difference of the selected discriminative feature
            # this tells us how good this teacher thinks it's discriminative feature is
            difference = teacher.difference(example, prediction, true_label, discriminative_feature[0]) 
            
            if(discriminative_feature in results):
                results[discriminative_feature] += difference
            else:
                results[discriminative_feature] = difference
             
        chosen_discriminative_feature = max(results, key=results.get) # get the best discriminative feature based on all the teachers
        return true_label, chosen_discriminative_feature
        