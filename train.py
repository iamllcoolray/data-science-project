import argparse
import yaml
import json
import pandas as pd
import logging
import os
import pickle

from datasets.cross_validation import get_dataset_split
from datasets.impute import impute_dataset
from datasets.preprocess import Preprocess

from models.model import create_model, generate_grid
from ensembles.ensemble_regression import create_ensemble as create_regression_ensemble

from metrics.regression_metrics import RegressionMetric

from utils.runs import create_run, dict_to_str
from utils.logger import create_global_logger, create_logger
import pprint

# TODO: Make logging prettier...
logger = create_global_logger(name = __name__, level = logging.DEBUG)

def main(config:dict, hyperparameters:dict):   
    data_cfg = config["DATA"] # everything data related is nested inside here 
    data_cfg["SEED"] = config["SEED"] 

    prep_config = config["PREPROCESSING"]
    prep_config["SEED"] = config["SEED"]

    data = pd.read_csv(data_cfg["DATA_PATH"])
    data = data[data_cfg["FEATURES"] + [data_cfg["TARGET"]]] # Keep only the features and target from the config

    data_features = data[data_cfg["FEATURES"]]
    data_target = data[[data_cfg["TARGET"]]] 

    
    data_features = impute_dataset(prep_config, data_features)
    
    train_indices, test_indices = get_dataset_split(data_cfg, data_features, data_target)

    # we save Preprocess in train because we need them for prediction to work (script to work). 
    prep_config["PREPROCESS_HYPERPARAMETERS"] = {}
    feat_prep = Preprocess(prep_config["FEATURES"], prep_config["PREPROCESS_HYPERPARAMETERS"])
    target_prep = Preprocess(prep_config["TARGET"])  


    result_cfg = config["RESULTS"]
    folder_path = None
    run_logger = None
    #if result_cfg["SAVE"]:
    folder_path = create_run(result_cfg)
    run_logger = create_logger(os.path.join(folder_path, "log.txt"), name = __name__)

    metric = RegressionMetric()

    run_logger.info("\n"+pprint.pformat(config))

    for algorithm_name in hyperparameters:
        # NOTE: Not using sklearn gridsearch to allow for custom pytorch gridsearch
        for hp_cross_product in generate_grid(hyperparameters[algorithm_name]):
            if config["MODEL"]["ENSEMBLE_TYPE"] is None or config["MODEL"]["ENSEMBLE_TYPE"] == "None":
                model = create_model(algorithm_name, hp_cross_product, seed=config["SEED"])
            else:
                model = create_regression_ensemble(config["MODEL"], algorithm_name, hp_cross_product, seed=config["SEED"])

            logger_msg = "\n"+"#"*50+"\n"+algorithm_name+"\n"+pprint.pformat(hp_cross_product)

            metric.reset() # Before starting on a new cross validation, reset the metric values.
            for fold_idx, (train_idx, test_idx) in enumerate(zip(train_indices, test_indices)):
                model.reset()
                X_train = data_features.iloc[train_idx]
                X_test = data_features.iloc[test_idx]

                y_train = data_target.iloc[train_idx]
                y_test = data_target.iloc[test_idx]

                # When doing train, do fit_transform for both x and y.
                # When doing test, only do it for x. 
                # In the loop below, notice how when the model predicts y, you need to inverse_transform to get it back on the same scale. 
                #X_train, y_train = feat_prep.fit_transform(X_train), target_prep.fit_transform(y_train)
                #X_test = feat_prep.transform(X_test)    
                X_train = feat_prep.fit_transform(X_train)
                X_test = feat_prep.transform(X_test)
                y_train = target_prep.fit_transform(y_train)           

                model.train(X_train, y_train)
                y_hat = target_prep.inverse_transform(model(X_test))
                #y_hat = model(X_test)
                #y_test = target_prep.transform(y_test)

                metric.update(y_test, y_hat)

                if result_cfg["VERBOSE"] >= 1:
                    results = metric(y_test, y_hat)
                    logger_msg += "\n\n"+"-"*50+"\nFOLD {0}\n".format(fold_idx+1)+pprint.pformat(results)+"\n"+"-"*50

                if result_cfg["SAVE"]:
                    fold_folder = "fold_" + str(fold_idx + 1)

                    file_name = algorithm_name + "_" + dict_to_str(hp_cross_product) + ".pkl"
                    with open(os.path.join(folder_path, fold_folder, file_name), "wb") as file:
                        pickle.dump(model, file)
    
                    feature_prep_path = os.path.join(folder_path, fold_folder, "feat_preprocess" + ".pkl")
                    target_prep_path = os.path.join(folder_path, fold_folder, "target_prep" + ".pkl")
                    # Only save the preprocessing once for each fold, since subsequent repetitions contain the same data.
                    if not os.path.exists(feature_prep_path):
                        with open(feature_prep_path, "wb") as file:
                            pickle.dump(feat_prep, file)
                    if not os.path.exists(target_prep_path):
                        with open(target_prep_path, "wb") as file:
                            pickle.dump(target_prep, file)
                    
            logger_msg += "\n\nMEAN\n"+pprint.pformat(metric.mean())+"\n\nSTDEV\n"+pprint.pformat(metric.stdev())+"\n"
            logger_msg += "#"*50
            run_logger.info(logger_msg)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "train.py",
         description = "Creates regression models on the given data."
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-p", "--hyperparameters", required=True)

    args = vars(parser.parse_args()) # takes everything passed thru console and parse using what we defined as args. the vars turns it into a dict. 

    config_path = args["config"] # args config is the key and whatever we pass in are the variables that go in. 
    hyperparameter_path = args["hyperparameters"] # args hyperparameters is the key and whatever we pass in are the variables that go in. 
    # the purpose of this is to prevent all the variables from being passed in thru console, making it extremely messy. 
    # this way, we just set them on the path to the config file that has them defined. 
    with open(config_path, "r") as file:
        try:
            # yaml isn't a safe format (ppl have gotten hacked), so we use safe load. if cp picks up anything weird, we'll get an error. 
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)
    # json does't have yaml safety issue, so we do normal loading. 
    with open(hyperparameter_path, "r") as file:
        hyperparameters = json.load(file)

    main(config, hyperparameters)