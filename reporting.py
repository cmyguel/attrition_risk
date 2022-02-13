import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from pathlib import Path
from diagnostics import model_predictions

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = Path(config['test_data_path']) 
model_path = Path(config['output_model_path'])


def plot_confusion_matrix(matrix):
    # original code: https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/

    ax = sns.heatmap(matrix, annot=True, cmap='YlGnBu')

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Real Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Return plot of the Confusion Matrix.
    return plt

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    # load test data
    test_data = pd.read_csv(test_data_path/"testdata.csv", low_memory=False)

    # prediction and confusion matrix
    preds = model_predictions(test_data)
    matrix = metrics.confusion_matrix(test_data['exited'], preds)

    # get plot from confusion matrix 
    plt = plot_confusion_matrix(matrix)

    # save plot 
    plt.savefig(model_path/'confusionmatrix.png')




if __name__ == '__main__':
    score_model()
