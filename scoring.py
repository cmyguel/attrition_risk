from curses.ascii import ETB
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path']) 
test_data_path = Path(config['test_data_path']) 
model_path = Path(config['output_model_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # take trained model
    with open(model_path/"trainedmodel.pkl", "rb") as f:
        pipeline = pickle.load(f)
    
    # load test data
    test_data = pd.read_csv(test_data_path/"testdata.csv", low_memory=False)

    # f1 score and accuracy
    y_pred = pipeline.predict(test_data)
    
    acc = accuracy_score( test_data['exited'], y_pred )
    # print("accuracy:", acc)

    f1 = f1_score( test_data['exited'], y_pred, pos_label=1 )
    # print("f1 score:", f1)

    # save f1 score
    with open(model_path/"latestscore.txt", 'w') as f:
        f.write(str(f1))
    
    return float(f1)

if __name__ == '__main__':
    score_model()
