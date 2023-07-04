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

    
if __name__ == "__main__":
    # ------------------- training -------------------
    
    # read the .data file into a Pandas DataFrame, specifying the delimiter
    dataset_name = 'zoo'
    # dataset = 'nursery'
    df = read_csv(f'{dataset_name}.data', delimiter=',')

    # extract input features (X) and target variable (y)
    X = df.iloc[:, :-1].values  # extract all columns except the last one as input features (X)
    y = df.iloc[:, -1].values   # extract the last column as the target variable (y)
    model_results = {}
    mean_results = []
    
    for optimized in [False, True]:
        for i in range(1, 4):
            if not optimized and i == 3:
                continue
            curr = []
            print(f"\nTeacher{i}:\n")
            for j in range(1, 6):
                print(f"run{j}")
                f_name = f"outputs/{dataset_name}_dataset/Teacher{i}_output{j}.txt"
                model = fm.feedbackModel(output_file_name=f_name, optimized=optimized)
                curr.append(tuple(model.fit(X, y, i)))
            
            model_results[(i, optimized)] = curr
         
    # ------------------- plotting -------------------
    
    for optimized in [False, True]:
        for i in range(1, 4):
            png_name = f"outputs/{dataset_name}_dataset/Teacher{i}_optimized={optimized}.png"

            if not optimized and i == 3:
                continue
            
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
            
   