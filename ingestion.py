import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = Path(config['input_folder_path'])
output_folder_path = Path(config['output_folder_path'])

#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    
    # get list of Paths of .csv files
    file_path_list = list(input_folder_path.glob("*.csv"))
    
    # read and concatenate dataframes
    df_finaldata = pd.concat(
        [pd.read_csv(input_path) for input_path in file_path_list]
        )
    
    # Drop duplicates and save dataframe
    df_finaldata.drop_duplicates(inplace=True)
    output_folder_path.mkdir(exist_ok=True)
    df_finaldata.to_csv(output_folder_path/'finaldata.csv', index=False)
    
    # save list of ingested files to "ingestedfiles.txt"    
    with open(output_folder_path/"ingestedfiles.txt", 'w') as f:
        f.write([str(path) for path in file_path_list].__str__())


if __name__ == '__main__':
    merge_multiple_dataframe()
    
    #NOTE: how to read
#     import ast
#     with open(output_folder_path/"ingestedfiles.txt", 'r') as f:
#         l = ast.literal_eval(f.read())
#         print(l)

 
