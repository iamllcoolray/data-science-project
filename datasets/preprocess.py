
# A list because we need to do preprocessing sequentially. For ex: we can do MinMaxScaler, followed by a normalizer to remove outliers. 
# We go through each name in list and if its there, we import its function and create an object. 
class Preprocess():
    import numpy as np
    def __init__(self, transformations:list, prep_config:dict = None) -> None:
        self.transformations = []
        for name in transformations:
            if name == "MinMaxScaler":
                from sklearn.preprocessing import MinMaxScaler
                self.transformations.append(MinMaxScaler())
            elif name == "StandardScaler":
                from sklearn.preprocessing import StandardScaler
                self.transformations.append(StandardScaler())


# same as sklearn inverse transform but in a list sequentially. 
# transform is for the testing sets (x)
    def transform(self, X):
        for transformation in self.transformations:
            X = transformation.transform(X)
        return X
# for training sets (x, and y)
    def fit_transform(self, X):
        for transformation in self.transformations:
            X = transformation.fit_transform(X)
        return X
    #  for test predictions outputted by model 
    def inverse_transform(self, X):
        for transformation in self.transformations:
            X = transformation.inverse_transform(X)
        return X