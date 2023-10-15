import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse

def custom_eval(test,predict)->dict:
    rmse = np.sqrt(mean_squared_error(test,predict))
    mae= mean_absolute_error(test,predict)
    r2= r2_score(test,predict)
    return {"RMSE": rmse,"RAE": mae, "R2 Accuracy": r2}

if __name__ == "__main__":
    #Set Experiment
    mlflow.set_experiment(experiment_name='CC Defaulter Experiment A')

    #Read and Split data
    data = pd.read_csv(r'C:/Users/KingX/OneDrive/Desktop/ML Projects/UCI_Credit_Card_Defaults.csv')
    train, test = train_test_split(data,test_size=0.25)
    train_x= train.drop(["ID","Defaulter"], axis=1)
    train_y= train[["Defaulter"]]
    test_x= test.drop(["ID","Defaulter"], axis=1)
    test_y= test[["Defaulter"]]

    #Input Parameters from user
    alp  = float(sys.argv[1]) if float(sys.argv[1])>1.0 else 0.5 
    l1r =  float(sys.argv[2]) if float(sys.argv[2])>2.0 else 0.5

    #Run experiment, select model and log details
    with mlflow.start_run():

        mdl = ElasticNet(alpha=alp, l1_ratio=l1r,random_state=10)
        
        mdl.fit(train_x,train_y)
        pred_y = mdl.predict(test_x)

        #performance = {}
        performance = custom_eval(test_y,pred_y)
        print("Parameters: alpha %f, l1-ratio %f" %(alp,l1r))
        print("Parameters: alpha {:f}, l1_ratio {:f}".format(alp,l1r))
        print(performance)

        #mlflow.log_param("alpha:",alp)
        mlflow.log_params(params={"alpha": alp,"l1_ratio": l1r})
        mlflow.log_metrics(metrics=performance)

        #Get remote model tracking registry
        remote_tracking_uri = urlparse(mlflow.get_tracking_uri()).scheme
        if remote_tracking_uri != "file":  #model registry does not work with file i.e. local file in local drive
            #Register the model
            mlflow.sklearn.log_model(mdl, "model", registered_model_name="ElasticNet Model", signature=infer_signature(train_x,mdl.predict(train_x)))
        else:
            #log in local uri or local machine
            mlflow.sklearn.log_model(mdl, "model", signature=infer_signature(train_x,mdl.predict(train_x)))










