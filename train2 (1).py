from __future__ import print_function

import argparse
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

## TODO: Import any additional libraries you need to define a model
from sklearn import svm

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    sklearn = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return sklearn


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
   
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    
    train_data = pd.read_csv('./train.csv', encoding='utf-8')

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    
    ## TODO: Define a model 
    sklearn = svm.SVC(gamma=0.001, kernel='linear', random_state=0)
        
    
    ## TODO: Train the model
    print('Training LR model')
    sklearn.fit(train_x, train_y)
    
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(sklearn, os.path.join(args.model_dir, "model.joblib"))