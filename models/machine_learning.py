


# Abstract class that takes in train and call method that we have to implement for all of our models 
class MachineLearning():
    def __init__(self, hyperparameters:dict):
        self.hyperparameters = hyperparameters

    def train(self, X, y):
        raise NotImplementedError("Subclass of Model should implement train")
    
    def __call__(self, X):
        raise NotImplementedError("Subclass of Model should implement __call__")
    
    def reset(self):
        raise NotImplementedError("Subclass of Model should implement reset")

class DecisionTree(MachineLearning):
    from sklearn.tree import DecisionTreeRegressor
    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        
        self.model = self.DecisionTreeRegressor(**self.hyperparameters)

    def train(self, X, y):
        self.model.fit(X, y)

    def __call__(self, X):
        return self.model.predict(X).reshape((-1,1))
    
    def reset(self):
        self.model = self.DecisionTreeRegressor(**self.hyperparameters)
    
class LinearRegression(MachineLearning):
    from sklearn.linear_model import LinearRegression
    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        

        self.model = self.LinearRegression(**self.hyperparameters) # passes the key values. when passing in fit_passing it'll read as fit_passing = True, instead of us writing out 

    def train(self, X, y):
        self.model.fit(X, y)

    def __call__(self, X):
        return self.model.predict(X)
    
    def reset(self):
        self.model = self.LinearRegression(**self.hyperparameters)

    
class GradientBoosting(MachineLearning):
    from sklearn.ensemble import GradientBoostingRegressor
    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

        self.model = self.GradientBoostingRegressor(**self.hyperparameters) # passes in the key values in hyp.json when model runs so we dont have to write out 

    def train(self, X, y):
        self.model.fit(X, y)
    
    def __call__(self, X):
        return self.model.predict(X).reshape(-1,1)
    
    def reset(self):
        self.model = self.GradientBoostingRegressor(**self.hyperparameters)


class ElasticNet(MachineLearning):
    import sklearn.linear_model as lm
    def __init__(self, hyperparameters:dict):
        super().__init__(hyperparameters)

        self.model = self.lm.ElasticNet(**self.hyperparameters)

    def train(self, X, y):
        self.model.fit(X, y)

    def __call__(self, X):
        return self.model.predict(X).reshape(-1,1)

    def reset(self):
        self.model = self.lm.ElasticNet(**self.hyperparameters)