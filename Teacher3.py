import numpy as np
from Teacher import Teacher
from Teacher1 import Teacher1
from Teacher2 import Teacher2
from random import choice, sample, randrange
# from TeacherWithSplittedData import Teacher


class Teacher3(Teacher):
    def __init__(self, X, labels, num_of_teachers=5, sample_percentage=0.6):
        # print(f"X.shape = {X.shape}\n")
        self.num_of_teachers = num_of_teachers  # number of teachers in the ensemble
        self.sample_percentage = sample_percentage  # percentage of samples to input to each teacher
        super().__init__(X, labels)
        self.teachers = self.initializeTeachers(labels)
        

    """
        @:returns  a list of num_of_teachers teachers by randomly sampling (without repetition) sample_percentage of data points
    """
    def initializeTeachers(self, labels): 
        # initialize a list where indexes_sampled_for_teachers[i] are the indexes of X, y we will input to teachers[i]
        indexes_sampled_for_teachers = [None for i in range(self.num_of_teachers)]
        
        all_indexes = range(self.X.shape[0])
        not_sampled_indexes = set(range(self.X.shape[0]))
        
        num_of_samples = int(self.X.shape[0] * self.sample_percentage)

        for i in range(self.num_of_teachers):
            sampled_indexes = list(sample(all_indexes, num_of_samples))
            # print(f"sampled_indexes = {sampled_indexes}\n")
            not_sampled_indexes -= set(sampled_indexes)
            # print(f"not_sampled_indexes = {not_sampled_indexes}\n")
            indexes_sampled_for_teachers[i] = sampled_indexes
                        
        # make sure that all indexes are selected for at least one teacher
        for i in not_sampled_indexes:
            selected_teachers = sample(range(self.num_of_teachers), randrange(1, self.num_of_teachers)) # randomly select a random number of teachers
            # print(f"selected_teachers = {selected_teachers}\n")
            for j in selected_teachers: # add the index i to the selected teachers
                # print(f"indexes_sampled_for_teachers[{j}] = {indexes_sampled_for_teachers[j]}\n")
                indexes_sampled_for_teachers[j].append(i)                
                # print(f"indexes_sampled_for_teachers[{j}] = {indexes_sampled_for_teachers[j]}\n")
            
        # print(f"X[indexes, :].shape = {self.X[indexes_sampled_for_teachers[0], :].shape}, labels[indexes].shape = {labels[indexes_sampled_for_teachers[0]].shape}\n")
        # print(f"X[np.array(indexes)].shape = {self.X[np.array(indexes_sampled_for_teachers[0])].shape}, labels[np.array(indexes)].shape = {labels[np.array(indexes_sampled_for_teachers[0])].shape}\n")
        teachers = [Teacher2(self.X[indexes, :], labels[indexes]) for indexes in indexes_sampled_for_teachers]
        # print(f"len(teachers) = {len(teachers)}\n")
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
        teachers_without_example = 0
        for teacher in self.teachers:            
            label, discriminative_feature = teacher.teach(example, explanation, prediction)
            if(label == None):
                teachers_without_example += 1
                continue
            # print(f"label = {label}\n")
            
            # get the score of the selected discriminative feature
            # this tells us how good this teacher thinks it's discriminative feature is
            df_score = teacher.discriminativeFeatureScore(example, prediction, true_label, discriminative_feature[0]) 
            
            if(discriminative_feature[0] in results):
                results[discriminative_feature[0]] += df_score
            else:
                results[discriminative_feature[0]] = df_score
        
        if not results:
            # print(f"teachers_without_example = {teachers_without_example}\n\nthis example failed:\n\nexample = {example}\nprediction = {prediction}\ntrue_label = {true_label}\nexplanation = {explanation}\n")
            for teacher in self.teachers:            
                label, discriminative_feature = teacher.teach(example, explanation, prediction)
                # print(f"label = {label}")
            # print(f"\ntrue_label = {true_label}\n")
            
             
        chosen_index = max(results, key=results.get) # get the best discriminative feature based on all the teachers
        return true_label, [chosen_index, example[chosen_index]]
        