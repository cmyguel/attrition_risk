import os
import shutil

# remove output folders 
shutil.rmtree('ingesteddata', ignore_errors=True)
shutil.rmtree('practicemodels', ignore_errors=True)

# use first configuration file
shutil.copy('config1.json', 'config.json')

# run all processes
os.system('python ingestion.py')
os.system('python training.py')
os.system('python scoring.py')
os.system('python deployment.py')
_ = input("run app.py in new terminal ans press Enter \n")
os.system('python apicalls.py')
os.system('python reporting.py')

# set second configuration file (to run fullprocess.py)
shutil.copy('config2.json', 'config.json')