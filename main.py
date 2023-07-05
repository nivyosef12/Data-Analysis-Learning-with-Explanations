import feedbackModel as fm
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

def plot(examples_seen, percent_mistakes, label=""):
    
    # if there is more than one figure, make sure we plot on the last one created
    last_figure = plt.gcf()
    plt.figure(last_figure.number)  # Set the current figure to the last one
    
    # create a line plot
    plt.plot(examples_seen, percent_mistakes, label=label)
 
   
def tune_teacher3(X, y):
    tune_results = {}
    mean_results = []
    params = [(num_of_teachers, sample_percentage) for num_of_teachers in [3, 5, 7] for sample_percentage in [0.6, 0.7, 0.8]] 
    
    for num_of_teachers, sample_percentage in params:
        curr = []
        print(f"\nTeacher3_optimized=True_with_params=({num_of_teachers}, {sample_percentage}):\n")
        for j in range(1, 6):
            print(f"tune run{j}")
            f_name = f"outputs/tune_reslts/Teacher3_optimized=True_with_params=({num_of_teachers}, {sample_percentage})_output{j}.txt"
            model = fm.feedbackModel(output_file_name=f_name, optimized=True)
            curr.append(tuple(model.fit(X, y, num_of_teachers, sample_percentage, 3)))
        
        tune_results[(num_of_teachers, sample_percentage)] = curr
         
    # ------------------- plotting -------------------
    for num_of_teachers, sample_percentage in params:
        png_name = f"outputs/tune_reslts/Teacher3_optimized=True_with_params=({num_of_teachers}, {sample_percentage}).png"
        
        curr_result = tune_results[num_of_teachers, sample_percentage]            
        for examples_seen, percent_mistakes in curr_result:
            plot(examples_seen, percent_mistakes)
        
           
        mean_results.append((curr_result[0][0],
                                np.mean(np.array([res[1] for res in curr_result]), axis=0),
                                f"Teacher3_optimized=True_with_params=({num_of_teachers}, {sample_percentage})"))
                
        # set the title and axis labels
        plt.title(f"Mistakes made over time of Teacher3_optimized=True_with_params=({num_of_teachers}, {sample_percentage})")
        plt.xlabel("Examples seen")
        plt.ylabel("Percentage of mistakes")
        
        # save as .png file
        plt.savefig(png_name)
        
        # create a figure for this teacher and optimized option
        plt.figure()  
        
        
    for examples_seen, percent_mistakes, label in mean_results:
        plot(examples_seen, percent_mistakes, label)
        
    
    # set the title and axis labels
    plt.title("Mistakes made over time of teachers")
    plt.xlabel("Examples seen")
    plt.ylabel("Percentage of mistakes")
    
    # add a legend to distinguish between graphs
    plt.legend()
    
    # save as .png file
    plt.savefig(f"outputs/tune_reslts/Avarage_mistakes_made.png")
    
    # create a figure for this teacher and optimized option
    plt.figure() 
    
        
def compareTeachers(X, y, optimized_lst, num_of_teachers, dataset_name):
    
    model_results = {}
    mean_results = []
    
    for optimized in optimized_lst:
        for i in range(1, num_of_teachers+1):
            
            curr = []
            print(f"\nTeacher{i}_optimized={optimized}:\n")
            for j in range(1, 6):
                print(f"run{j}")
                f_name = f"outputs/{dataset_name}_dataset/Teacher{i}_optimized={optimized}_output{j}.txt"
                model = fm.feedbackModel(output_file_name=f_name, optimized=optimized)
                curr.append(tuple(model.fit(X, y, i)))
            
            model_results[(i, optimized)] = curr
         
    # ------------------- plotting -------------------
    
    for optimized in optimized_lst:
        for i in range(1, num_of_teachers+1):
            png_name = f"outputs/{dataset_name}_dataset/Teacher{i}_optimized={optimized}.png"
            
            curr_result = model_results[i, optimized]            
            for examples_seen, percent_mistakes in curr_result:
                plot(examples_seen, percent_mistakes)
                
            mean_results.append((curr_result[0][0],
                                 np.mean(np.array([res[1] for res in curr_result]), axis=0),
                                 f"Teacher{i}_optimized={optimized}"))
                
            # set the title and axis labels
            plt.title(f"Mistakes made over time of Teacher{i}_optimized={optimized}")
            plt.xlabel("Examples seen")
            plt.ylabel("Percentage of mistakes")
            
            # save as .png file
            plt.savefig(png_name)
            
            # create a figure for this teacher and optimized option
            plt.figure()  
            
    for examples_seen, percent_mistakes, label in mean_results:
        plot(examples_seen, percent_mistakes, label)
        
    
    # set the title and axis labels
    plt.title("Mistakes made over time of teachers")
    plt.xlabel("Examples seen")
    plt.ylabel("Percentage of mistakes")
    
    # add a legend to distinguish between graphs
    plt.legend()
    
    # save as .png file
    plt.savefig(f"outputs/{dataset_name}_dataset/Avarage_mistakes_made.png")
    
    # create a figure for this teacher and optimized option
    plt.figure() 

    
if __name__ == "__main__":
    # # ------------------- Part A -------------------
    # print("-----------------------Part A-------------------------\n\n")
    
    # print("Zoo:\n")
    
    # # read the .data file into a Pandas DataFrame, specifying the delimiter
    # dataset_name = 'zoo'
    # df = read_csv(f'{dataset_name}.data', delimiter=',')

    # # extract input features (X) and target variable (y)
    # X = df.iloc[:, :-1].values  # extract all columns except the last one as input features (X)
    # y = df.iloc[:, -1].values   # extract the last column as the target variable (y)
    
    # compareTeachers(X, y, [False], 2, dataset_name)
    
    # print("Nursery:\n")
    
    # dataset_name = 'nursery'
    # df = read_csv(f'{dataset_name}.data', delimiter=',')

    # # extract input features (X) and target variable (y)
    # X = df.iloc[:, :-1].values  # extract all columns except the last one as input features (X)
    # y = df.iloc[:, -1].values   # extract the last column as the target variable (y)
    
    # compareTeachers(X, y, [False], 2, dataset_name)


    # ------------------- Part B -------------------

    print("-----------------------Part B-------------------------\n\n")

    # print("online_shoppers_intention:\n")
    
    # dataset_name = 'online_shoppers_intention'
    # df = read_csv(f'{dataset_name}.csv', delimiter=',')

    
    dataset_name = 'wifi_localization'
    df = read_csv(f'datasets/{dataset_name}.txt', delimiter='\t', header=None)
    

    # extract input features (X) and target variable (y)
    X = df.iloc[:, :-1].values  # extract all columns except the last one as input features (X)
    y = df.iloc[:, -1].values   # extract the last column as the target variable (y)
       
   
    
    # print("before for")
    # for i in range(y.shape[0]):
    #     print(f"i = {i}\n")
    #     for j in range(i+1, y.shape[0]):
    #         print(f"j = {j}\n")
    #         if np.array_equal(X[i], X[j]) and y[i] != y[j]:
    #             print(f"X[{i}] = X[{j}], y[{i}] = {y[i]} and y[{j}] = {y[j]}\n")
    
    # print("no problem")
    
    compareTeachers(X, y, [True, False], 3, dataset_name)
    # tune_teacher3(X, y)