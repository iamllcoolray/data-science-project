import pandas as pd

# instead of having a lot of objects defined in train, define them in the modules to make it easier to debug, and changing logic. 
def get_dataset_split(cfg:dict, X:pd.DataFrame, y:pd.DataFrame):
    if cfg["SPLIT_TYPE"] == "Random":
        return RandomFold(X, y, n_splits=5, shuffle=True, seed=cfg["SEED"]).split()
    elif cfg["SPLIT_TYPE"] == "Stratified":
        return StratifiedFold(X, y, n_splits=5, num_quantiles=10, shuffle=True, seed=cfg["SEED"]).split()
    return None

# Abstract class where all children inherit from 
class CrossValidation():
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame) -> None:
        self.X = X
        self.y = y

    
    def split(self): # everyone has split() defined, modify for each split implementation
        raise NotImplementedError("Subclass of Split should implement split")
        # Return indices for training, indices for testing
    
class RandomFold(CrossValidation):
    from sklearn import model_selection # keep imports spceific to their methods to keep imports section clean and efficiency 
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame, n_splits:int = 5, shuffle:bool = True, seed:int = 1337) -> None:
        super().__init__(X, y)

        self.cross_validator = self.model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    def split(self): # to use the imports inside the class, you have to use self. 
        # always include the module where the function resides, just in case we implement our own train_test_split (or own function)
        # Always use seed for results reproducibility 
        generator = self.cross_validator.split(self.X, self.y) # You can only iterate through a generator once...
        train_indices, test_indices = zip(*list(generator))
        return train_indices, test_indices
    

class StratifiedFold(CrossValidation):
    from sklearn import model_selection
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame, n_splits:int = 5, num_quantiles:int = 10,
            shuffle:bool = True, seed:int = 1337) -> None:
        self.X = X
        #self.y = y

        labels = [str(q) for q in range(num_quantiles)]
        # creating the bins (10 and then name them)
        self.discrete_y = pd.qcut(y.T.squeeze(axis=0).rank(method="first"), q=num_quantiles, labels=labels) # Tranpose y, then remove outer (zeroth) dim

        self.cross_validator = self.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    def split(self):
        generator = self.cross_validator.split(self.X, self.discrete_y) # You can only iterate through a generator once...
        train_indices, test_indices = zip(*list(generator))
        return train_indices, test_indices

    
        


