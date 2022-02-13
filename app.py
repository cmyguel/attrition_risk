from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis 
# import predict_exited_from_saved_model
import json
import os

from pathlib import Path
from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, dataframe_missing_data, execution_time, outdated_packages_list


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    #add return value for prediction outputs
    file_path = Path(request.args.get("file_path")).resolve()
    data = pd.read_csv(file_path, low_memory=False)
    preds = model_predictions(data)
    return jsonify({"prediction": preds})

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    #add return value (a single F1 score number)
    return jsonify({"f1_score": score_model()})

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    #check means, medians, and modes for each column
    #return a list of all calculated summary statistics
    return jsonify({"stats": dataframe_summary()})

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    #add return value for all diagnostics
    return jsonify({
        "timing": execution_time(),
        "missing_data": dataframe_missing_data(),
        "dependency_check": outdated_packages_list().to_json(),

    })

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
