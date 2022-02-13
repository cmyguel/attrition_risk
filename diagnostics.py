
# from copyreg import pickle
import pandas as pd
import numpy as np
import timeit
import os
import json

from pathlib import Path
import pickle
import timeit
import subprocess
from io import StringIO

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = Path(config['output_folder_path']) 
test_data_path = Path(config['test_data_path']) 
prod_deployment_path = Path(config['prod_deployment_path']) 

##################Function to get model predictions
def model_predictions(dataset):
    #read the deployed model and a test dataset, calculate predictions
    #return value should be a list containing all predictions
    with open(prod_deployment_path/"trainedmodel.pkl", 'rb') as f:
        pipeline = pickle.load(f)
    preds = pipeline.predict(dataset)
    return preds.tolist()

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here: mean, median, std
    #return value should be a list containing all summary statistics
    finaldata = pd.read_csv(dataset_csv_path/"finaldata.csv", low_memory=False)
    stats_list = []

    for col in finaldata.select_dtypes(include=np.number).columns:
        stats_list.append(np.mean(finaldata[col]))
        stats_list.append(np.median(finaldata[col]))
        stats_list.append(np.std(finaldata[col]))
    
    # convert np.float64 to float
    stats_list = [float(x) for x in stats_list]
    return stats_list

##################Function to get missing data
def dataframe_missing_data():
    #returns a list of percentage of missing data for each column in the data
    finaldata = pd.read_csv(dataset_csv_path/"finaldata.csv", low_memory=False)

    percent_missing = finaldata.isnull().sum() * 100 / len(finaldata)
    return [float(x) for x in percent_missing.values]

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    #return a list of 2 timing values in seconds
    delay_list = []

    t_1 = timeit.default_timer()
    os.system('python3 ingestion.py')
    dt = timeit.default_timer() - t_1
    delay_list.append(dt)

    t_1 = timeit.default_timer()
    os.system('python3 training.py')
    dt = timeit.default_timer() - t_1
    delay_list.append(dt)

    return delay_list

##################Function to check dependencies
def outdated_packages_list():
    #get outdated packages
    bytes_data = subprocess.check_output(['pip', 'list','--outdated'])

    #convert bytes response to dataframe
    s=str(bytes_data,'utf-8')
    data = StringIO(s) 
    df=pd.read_fwf(data)

    #drop Type column and first row that contains no information
    df.drop("Type", axis=1, inplace=True)
    df.drop(0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

if __name__ == '__main__':
    test_data = pd.read_csv(test_data_path/"testdata.csv", low_memory=False)

    # print(model_predictions(test_data))
    # print(dataframe_summary())
    # print(dataframe_missing_data())
    # print(execution_time())
    print(outdated_packages_list())
