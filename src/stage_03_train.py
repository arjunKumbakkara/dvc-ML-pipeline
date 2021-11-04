from src.utils.all_utils import create_directory, read_yaml, save_local_df
import argparse
import pandas as pd
#import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import joblib

def train(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts_dir= config["artifacts"]['artifacts_dir']
    split_data_dir= config["artifacts"]["split_data_dir"]
    
    train_data_filename= config["artifacts"]["train"]
    #test_data_filename= config["artifacts"]["test"]


    train_data_path= os.path.join(artifacts_dir,split_data_dir,train_data_filename)
    #test_data_path= os.path.join(artifacts_dir,split_data_dir,test_data_filename)

    train_data = pd.read_csv(train_data_path)


    #Get X and Y Values
    train_y= train_data["quality"] 
    train_x= train_data.drop("quality",axis=1)

    #Getting Params from params.yaml
    alpha= params["model_params"]["ElasticNet"]["alpha"]
    l1_ratio= params["model_params"]["ElasticNet"]["l1_ratio"]
    random_state= params["base"]["random_state"]
    
        



    lr=ElasticNet(alpha=0.5,l1_ratio=0.5,random_state=42)
    lr.fit(train_x,train_y)

    model_dir=config["artifacts"]["model_dir"]
    model_filename=config["artifacts"]["model_filename"]

    model_dir=os.path.join(artifacts_dir,model_dir)
    create_directory([model_dir])    
    
    model_path=os.path.join(model_dir, model_filename)


    joblib.dump(lr,model_path)




if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--params","-p",default="params.yaml")
    
    parsed_args =args.parse_args()


    train(config_path=parsed_args.config,params_path = parsed_args.params)
