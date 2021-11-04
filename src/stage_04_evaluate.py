from src.utils.all_utils import create_directory, read_yaml, save_local_df, save_reports
import argparse
import pandas as pd
#import sys
import os
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import ElasticNet
import joblib
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np


def evaluate_metrics(actual_value,predicted_values):
    rmse= np.sqrt(mean_squared_error(actual_value, predicted_values))
    mae= mean_absolute_error(actual_value, predicted_values)
    r2= r2_score(actual_value, predicted_values)

    return rmse,mae,r2


def evaluate(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    #Artifacts is the base
    artifacts_dir= config["artifacts"]['artifacts_dir']
          
    #We need the test data for evaluation
    test_data_filename= config["artifacts"]["test"]
    split_data_dir= config["artifacts"]["split_data_dir"]
    test_data_path= os.path.join(artifacts_dir,split_data_dir,test_data_filename)

    #Get X and Y Values for the test data for evaluation 
    test_data=pd.read_csv(test_data_path)
    test_y= test_data["quality"] 
    test_x= test_data.drop("quality",axis=1)


    model_dir=config["artifacts"]["model_dir"]
    model_filename=config["artifacts"]["model_filename"]
    model_path=os.path.join(artifacts_dir,model_dir, model_filename)
    #loading the model
    lr=joblib.load(model_path)

    #While evluation, we need to  first predict and then evaluate them against the ground truth
    predicted_values = lr.predict(test_x )
    rmse,mae,r2= evaluate_metrics(test_y, predicted_values)
    print("\n")
    print(f"rmse(mean squared Error) --> {rmse} : mae(mean absolute error)--> {mae} : r2(r2 square)-->{r2} ")

    #Getting the scores path and scores file name for saving this into a collection for future comparisons. 
    scores_dir=config["artifacts"]["reports_dir"]
    scores_file=config["artifacts"]["scores"]
    
    create_directory([os.path.join(artifacts_dir,scores_dir)])
    scores_path=os.path.join(artifacts_dir,scores_dir,scores_file)

    #For Individual Report
    for report,reportName in {'SquaredError':{'RMSE Value':rmse},'meanError':{'MAE Value':rmse},'r2Score':{'R2 Value':rmse}}.items():
        save_reports(reportName,scores_path.split('.')[0]+"_"+report+".json")
        # print(f" Report: {reportName} is saved @ {scores_path}")

    #For Consolidated Report
    save_reports({'SquaredError':rmse,'meanError':mae,'r2Score':r2},scores_path.split('.')[0]+"_"+"CONSOLIDATED"+".json")
        # print(f" Report: {reportName} is saved @ {scores_path}")






if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--params","-p",default="params.yaml")
    
    parsed_args =args.parse_args()


    evaluate(config_path=parsed_args.config,params_path = parsed_args.params)
