import re
import requests
import json
from pathlib import Path

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = Path(config['output_model_path']) 


#Call each API endpoint and store the responses
response1 = requests.post(URL+'/prediction?file_path=testdata/testdata.csv')
response2 = requests.get(URL+'/scoring')
response3 = requests.get(URL+'/summarystats')
response4 = requests.get(URL+'/diagnostics')

#combine all API responses
responses = [response1.json(), response2.json(), response3.json(), response4.json()]

#write the responses to your workspace
with open(model_path/"apireturns.txt", 'w') as f:
    f.write(responses.__str__())

#print responses
# if __name__ == '__main__':

#     print("/prediction")
#     print(response1)
#     print(response1.json())

#     print("/scoring")
#     print(response2)
#     print(response2.json())

#     print("/summarystats")
#     print(response3)
#     print(response3.json())

#     print("/diagnostics")
#     print(response4)
#     print(response4.json())

