import pandas as pd
import numpy as np

# based on the config here, we make the objects here and work with them.
# so this way, train.py stays clean. we dont have to make a lot of 'if' statements, and so forth... 
def impute_dataset(config:dict, dataset:pd.DataFrame):
    if config["IMPUTE_TYPE"] is None:
        return Constant(constant=-1).impute(dataset)
    elif config["IMPUTE_TYPE"] == "MICE":
        return MICE(NaN_value=-1.0, seed=config["SEED"]).impute(dataset)

 # abstract class with all of the imputations we're doing 
class Impute():
    def __init__(self) -> None: # takes in itself and returns nothing 
        pass

    def impute(self, X:pd.DataFrame): 
        raise NotImplementedError("Subclass of Impute should implement impute")
    
    # rn there isn't really any imputation going on, just -1 for missing values 
    # Any imputation completed in the future has to use the same impute() method 
    # when we do MICE it'll inherent standard impute() method from parent, and then we'll modify it to make it MICE in the child class 
    # Any other kind of imputation we do, we make a new child class 
class Constant(Impute): # inherits Impute abstract class 
    def __init__(self, constant:int = -1) -> None: # -1 bc altemis dataset has -1 for missing values 
        super().__init__() # initializes parent class 
        self.constant = constant

    def impute(self, X:pd.DataFrame):
        X = X.fillna(self.constant)
        return X

class MICE(Impute): 
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import LinearRegression
    def __init__(self, NaN_value:float = -1.0, seed:int = 1337): # if not passing anything, then it'll be -1 for default. 
        super().__init__()
        self.NaN_value = NaN_value

        lr = self.LinearRegression()
        self.imp = self.IterativeImputer(estimator = lr, verbose = 0, max_iter =  70, tol = 1e-3, 
            imputation_order='random', random_state=seed) 
    
    def impute(self, X:pd.DataFrame):
        X = X.replace({self.NaN_value: np.NaN})
        X_imputed = self.imp.fit_transform(X)
        imputed_df = pd.DataFrame(X_imputed, columns = X.columns)
        return imputed_df