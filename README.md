# README

## Environment Configuration

To configure the environment, install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Running the Source Code

To execute the training process, use the following command:

```bash
python train.py --config workflow_config.yaml --hyperparameters base_model_hyperparameters.json
```

Replace `workflow_config.yaml` and `base_model_hyperparameters.json` with the desired configuration and hyperparameter files, where `workflow_config.yaml` determines worflow parameters like dataset path, features, target, what preprocessing algorithms to use (if any), what ensemble algorithms to use (if any), etc. and `base_model_hyperparameters.json` determines the grid search hyperparameters to use for a given machine learning or deep learning model.


For hypeparameter optimization of bagging and boosting algorithms, use the shell scripts in the `shell_scripts` folders that correspond to your OS.

**Windows**
```bash
shell_scripts\run_bagging_configs.bat
```

**Linux**
```bash
./shell_scripts/run_bagging_configs.sh
```

After models have finished running, use the following command to get lowest MAE and SD results across all folds: 

```bash
python stand_alone_scripts/find_lowest_sd.py  
python stand_alone_scripts/find_lowest_mae.py 
```
---

## Example Configurations and Hyperparameters

### Base Model

```yaml
DATA:
  DATA_PATH: heart.csv
  FEATURES: ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
  PROBLEM: Regression
  SPLIT_TYPE: Stratified
  TARGET: target
MODEL:
  ENSEMBLE_HYPERPARAMETERS:
    LEARNING_RATE: 0.001
    LOSS: MSE
    NUM_BOOSTRAP_SAMPLES: 5
    NUM_ESTIMATORS: 5
    NUM_THETA_INTERVALS: 2
    P: 2
  ENSEMBLE_TYPE: null
PREPROCESSING:
  FEATURES:
  - StandardScaler
  IMPUTE_TYPE: MICE
  TARGET:
  - StandardScaler
RESULTS:
  OUTPUT_DIR: runs
  SAVE: true
  VERBOSE: 1
SEED: 1337
```

### Bagging

```yaml
DATA:
  DATA_PATH: heart.csv
  FEATURES: ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
  PROBLEM: Regression
  SPLIT_TYPE: Stratified
  TARGET: target
MODEL:
  ENSEMBLE_HYPERPARAMETERS:
    LEARNING_RATE: 0.001
    LOSS: MSE
    NUM_BOOSTRAP_SAMPLES: 5
    NUM_ESTIMATORS: 5
  ENSEMBLE_TYPE: Bagging
PREPROCESSING:
  FEATURES:
  - StandardScaler
  IMPUTE_TYPE: MICE
  TARGET:
  - StandardScaler
RESULTS:
  OUTPUT_DIR: runs
  SAVE: true
  VERBOSE: 1
SEED: 1337
```

This configuration uses the following parameters:
- `NUM_BOOSTRAP_SAMPLES`: Controls the number of boostrap samples to use.

### Boosting

```yaml
DATA:
  DATA_PATH: heart.csv
  FEATURES: ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
  PROBLEM: Regression
  SPLIT_TYPE: Stratified
  TARGET: target
MODEL:
  ENSEMBLE_HYPERPARAMETERS:
    LEARNING_RATE: 0.001
    LOSS: MSE
    NUM_BOOSTRAP_SAMPLES: 5
    NUM_ESTIMATORS: 5
  ENSEMBLE_TYPE: Boosting
PREPROCESSING:
  FEATURES:
  - StandardScaler
  IMPUTE_TYPE: MICE
  TARGET:
  - StandardScaler
RESULTS:
  OUTPUT_DIR: runs
  SAVE: true
  VERBOSE: 1
SEED: 1337
```

This configuration uses the following parameters:
- `LEARNING_RATE`: Controls the learning rate of each weak learner.
- `LOSS`: Determines the optimization function used when combining weak learner and global learner predictions.
- `NUM_ESTIMATORS`: Determines the number of weak learners to use.


### Decision Tree JSON Hyperparameters

```json
{
    "Decision Tree": {
        "criterion": ["absolute_error"],
        "splitter": ["best"],
        "max_depth": [10],
        "min_samples_split": [0.1],
        "max_features": ["log2"]
    }

}
```

This configuration uses the given hyperparameters for the chosen algorithm.




