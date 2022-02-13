import training
import scoring
import deployment
import diagnostics
import reporting

import os
from pathlib import Path
import pickle
import json
import ast
import pandas as pd
from sklearn.metrics import f1_score


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = Path(config['input_folder_path'])
output_folder_path = Path(config['output_folder_path'])
model_path = Path(config['output_model_path']) 
prod_deployment_path = Path(config['prod_deployment_path']) 

##################Check and read new data
#first, read ingestedfiles.txt
with open(output_folder_path/"ingestedfiles.txt", 'r') as f:
    prev_ingested_files = ast.literal_eval(f.read())

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
print("previously ingested files:", prev_ingested_files)
current_files = {str(file) for file in input_folder_path.glob("*.csv")}
new_files = current_files.difference(prev_ingested_files)

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if len(new_files)>0:
    print(f"{new_files} are not one of the previously ingested files {prev_ingested_files}. Running ingestion.py")
    os.system('python3 ingestion.py')
else:
    print("there are no new files to ingest. Process finished.")
    exit()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(prod_deployment_path/"latestscore.txt", "r") as f:
    prev_f1 = float(f.read())

with open(prod_deployment_path/"trainedmodel.pkl", "rb") as f:
    pipeline = pickle.load(f)

new_data = pd.read_csv(output_folder_path/"finaldata.csv", low_memory=False)
y_pred = pipeline.predict(new_data)
new_f1 = f1_score( new_data['exited'], y_pred, pos_label=1 )


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if prev_f1 > new_f1:
    print(f"there is evidence of model drift (prev_f1:{prev_f1:.2f}, new_f1:{new_f1:.2f}). Proceeding to re-deployment.")
else:
    print(f"There are no signs of model drift (prev_f1:{prev_f1:.2f}, new_f1:{new_f1:.2f}). Process finished.")
    exit()

##################Re-training
os.system('python training.py')
os.system('python scoring.py')

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
os.system('python deployment.py')

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system('python apicalls.py')
os.system('python reporting.py')







