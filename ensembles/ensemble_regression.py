from models.model import create_model

def create_ensemble(config:dict, algorithm_name:str, hyperparameters:dict, seed:int):
    ensemble_config = config["ENSEMBLE_HYPERPARAMETERS"]
    if config["ENSEMBLE_TYPE"] == "Bagging":
        return BaggingEnsemble(ensemble_config["NUM_BOOSTRAP_SAMPLES"], create_model, algorithm_name, hyperparameters, seed)
    elif config["ENSEMBLE_TYPE"] == "Boosting":
        return BoostingEnsemble(ensemble_config["NUM_ESTIMATORS"], ensemble_config["LEARNING_RATE"], ensemble_config["LOSS"],
            create_model, algorithm_name, hyperparameters, seed)

class BaggingEnsemble:
    from typing import Callable
    import numpy as np
    from sklearn import utils as sk_utils
    def __init__(self, num_bagging_samples:int, create_model:Callable, algorithm_name:str, hyperparameters:dict, seed:int):
        self.create_model = create_model
        self.algorithm_name = algorithm_name
        self.hyperparameters = hyperparameters
        self.seed = seed
        self.num_bagging_samples = num_bagging_samples

        self.models = []

    def train(self, X:np.ndarray, y:np.ndarray):
        for _ in range(self.num_bagging_samples):
            X_resample, y_resample = self.sk_utils.resample(X, y)

            base_model = self.create_model(self.algorithm_name, self.hyperparameters, self.seed)
            base_model.train(X_resample, y_resample)

            self.models.append(base_model)
    
    def __call__(self, X:np.ndarray):
        y_hat = 0
        for base_model in self.models:
            y_hat += base_model(X).reshape((-1,1))
        
        return y_hat / len(self.models)
    
    def reset(self):
        self.models = []


class BoostingEnsemble:
    from typing import Callable
    import numpy as np
    def __init__(self, num_estimators:int, learning_rate:float, loss:str, create_model:Callable, algorithm_name:str, hyperparameters:dict, seed:int):
        self.num_estimators = num_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.create_model = create_model
        self.algorithm_name = algorithm_name
        self.hyperparameters = hyperparameters
        self.seed = seed

        self.models = []
        self.initial_prediction = None

    def compute_residuals(self, y, y_hat):
        # NOTE: These are negative gradient of Loss(y, y_hat) with respect to y_hat, which for MSE is just y - y_hat
        if self.loss == "MSE":
            return y - y_hat 

    def train(self, X:np.ndarray, y:np.ndarray):
        self.initial_prediction = self.np.mean(y)
        y_hat = self.np.full_like(y, self.initial_prediction, dtype=self.np.float32)

        for _ in range(self.num_estimators):
            residuals = self.compute_residuals(y, y_hat)

            base_model = self.create_model(self.algorithm_name, self.hyperparameters, self.seed)
            base_model.train(X, residuals)

            self.models.append(base_model)

            y_hat += self.learning_rate * base_model(X)

    def __call__(self, X:np.ndarray):
        y_hat = self.np.full(X.shape[0], self.initial_prediction, dtype=self.np.float32).reshape((-1,1))

        for base_model in self.models:
            y_hat += self.learning_rate * base_model(X)

        return y_hat

    def reset(self):
        self.models = []