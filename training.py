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
from sklearn.metrics import accuracy_score, f1_score

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path']) 
model_path = Path(config['output_model_path']) 


#################Function for training the model
def train_model():
    #define columns used for inference
    inference_cols = [  "lastmonth_activity",
                        "lastyear_activity", 
                        "number_of_employees",
                        ]
    
    #use this logistic regression for training
    log_reg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False,
                    )
    
    pipeline = Pipeline([
    #selects only inferenfe columns
    ("selector", ColumnTransformer([ ("selector", "passthrough", inference_cols) ], remainder="drop")),
    # scales inputs
    # ("scaler", StandardScaler()),
    #logistic regression classifier
    ("classifier", log_reg),
    ])

    #fit the logistic regression to your data
    data = pd.read_csv(dataset_csv_path/"finaldata.csv", low_memory=False)
    pipeline.fit(data, data["exited"])
  
    #create folder if unexistent 
    #write the trained model to your workspace in a file called trainedmodel.pkl
    model_path.mkdir(exist_ok=True)
    with open(model_path/"trainedmodel.pkl", "wb") as f:
        pickle.dump( pipeline, f )

if __name__ == '__main__':
    train_model()