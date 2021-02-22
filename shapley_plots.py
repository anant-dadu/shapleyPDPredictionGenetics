import pandas as pd
import xgboost as xgb
import numpy as np
import shap
import logging
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import pickle 
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s [%(levelname)s] %(message)s",
    format='%(asctime)s %(message)s', 
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

logger=logging.getLogger() 
logger.setLevel(logging.INFO) 
# logger.debug("Harmless debug Message") 
# logger.info("Just an information") 
# logger.warning("Its a Warning") 
# logger.error("Did you try to divide by zero") 
# logger.critical("Internet is down") 

class ShapleyExplainations:

    def __init__(self):
        pass

    def trainXGBModel(self, data, feature_names, label_name):
        logger.info('Training starts...')
        X_train, X_valid, y_train, y_valid = train_test_split(data, data[label_name] , test_size=0.3, random_state=42)
        ID_train = X_train['ID']
        ID_test = X_valid['ID']
        X_train = X_train[feature_names].copy()
        X_valid = X_valid[feature_names].copy()
        num_round = 500
        param = {
            'objective':'binary:logistic',
            "eta": 0.05,
            "max_depth": 10,
            "tree_method": "gpu_hist",
        }
        # GPU accelerated training
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
        logger.info('Data loaded! Model training starts...')
        model = xgb.train(param, dtrain,num_round)
        logger.info('Model trained! Shap Values Prediction starts...')
        model.set_param({"predictor": "gpu_predictor"})
        explainer_train = shap.TreeExplainer(model)
        shap_values_train = explainer_train.shap_values(X_train)
        explainer_test = shap.TreeExplainer(model)
        shap_values_test = explainer_test.shap_values(X_valid)
        logger.info('Training Completed!!!')
        dataset = {'X_train': X_train, 'X_valid': X_valid, 'y_train': y_train, 'y_valid': y_valid}
        shap_values = {'shap_values_train': shap_values_train, 'shap_values_test': shap_values_test, }
        #                'shap_values_interaction_train': shap_values_interaction_train, 'shap_values_interaction_test': shap_values_interaction_test}
        y_pred_test = model.predict(dtest)
        y_pred_train = model.predict(dtrain)
        # print ('Accuracy Test, Train:', np.sum(y_valid==y_pred_test.round()) / len(y_pred_test), np.sum(y_train==y_pred_train.round()) / len(y_pred_train))
        print ('Auc Score Test, Train:', roc_auc_score(y_valid, y_pred_test), roc_auc_score(y_train, y_pred_train))
        other_info = {}
        other_info['ID_train'] = ID_train
        other_info['ID_test'] = ID_test
        other_info['y_pred_test'] = y_pred_test
        other_info['y_pred_train'] = y_pred_train
        other_info['y_test'] = y_valid
        other_info['y_train'] = y_train
        other_info['AUC_train'] = roc_auc_score(y_train, y_pred_train)
        other_info['AUC_test'] = roc_auc_score(y_valid, y_pred_test)

        # explainer_model = shap.Explainer(model, X)
        # shap_values_model = explainer_model(X)
        expected_values = {'explainer_train': explainer_train.expected_value,  'explainer_test': explainer_test.expected_value}
        return shap_values, dataset, expected_values, other_info
    
    def trainXGBModel_cpu(self, data, feature_names, label_name):
        logger.info('Training starts...')
        X_train, X_valid, y_train, y_valid = train_test_split(data, data[label_name] , test_size=0.3, random_state=42)
        ID_train = X_train['ID']
        ID_test = X_valid['ID']
        X_train = X_train[feature_names].copy()
        X_valid = X_valid[feature_names].copy()
        num_round = 500
        param = {
            'objective':'binary:logistic',
            "eta": 0.05,
            "max_depth": 10,
            "tree_method": "gpu_hist",
        }
        # GPU accelerated training

        logger.info('Data loaded! Model training starts...')
        model = xgb.XGBClassifier().fit(X_train, y_train)
        explainer_train = shap.Explainer(model, X_train)
        shap_values_train = explainer_train(X_train)  

        explainer_test = shap.Explainer(model, X_valid)
        shap_values_test = explainer_test(X_valid)  

        explainer = {'explainer_train': explainer_train, 'explainer_test': explainer_test, } 
        # dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        # dtest = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
        logger.info('Model trained! Shap Values Prediction starts...')
        
        
        logger.info('Training Completed!!!')
        dataset = {'X_train': X_train, 'X_valid': X_valid, 'y_train': y_train, 'y_valid': y_valid}
        shap_values = {'shap_values_train': shap_values_train, 'shap_values_test': shap_values_test, }
        #                'shap_values_interaction_train': shap_values_interaction_train, 'shap_values_interaction_test': shap_values_interaction_test}
        # y_pred_test = model.predict(dtest)
        y_pred_test = model.predict_proba(X_valid)[:, 1]
        # y_pred_train = model.predict(dtrain)
        y_pred_train = model.predict_proba(X_train)[:, 1] 
        # print ('Accuracy Test, Train:', np.sum(y_valid==y_pred_test.round()) / len(y_pred_test), np.sum(y_train==y_pred_train.round()) / len(y_pred_train))
        print ('Auc Score Test, Train:', roc_auc_score(y_valid, y_pred_test), roc_auc_score(y_train, y_pred_train))
        other_info = {}
        other_info['ID_train'] = ID_train
        other_info['ID_test'] = ID_test
        other_info['y_pred_test'] = y_pred_test
        other_info['y_pred_train'] = y_pred_train
        other_info['y_test'] = y_valid
        other_info['y_train'] = y_train
        other_info['AUC_train'] = roc_auc_score(y_train, y_pred_train)
        other_info['AUC_test'] = roc_auc_score(y_valid, y_pred_test)

        # explainer_model = shap.Explainer(model, X)
        # shap_values_model = explainer_model(X)

        return shap_values, dataset, explainer, other_info