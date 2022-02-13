from re import S
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
import shutil



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path']) 
prod_deployment_path = Path(config['prod_deployment_path']) 
model_path = Path(config['output_model_path']) 


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    prod_deployment_path.mkdir(exist_ok=True)
    for src_path in [   model_path/"trainedmodel.pkl",
                        model_path/"latestscore.txt",
                        dataset_csv_path/"ingestedfiles.txt" ]:
        
        if src_path.exists():
            shutil.copy(src_path, prod_deployment_path)
        else:
            raise Exception("missing file: "+ str(src_path))

if __name__ == '__main__':
    store_model_into_pickle()
        
        
        

