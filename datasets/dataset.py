from torch.utils.data import Dataset
import pandas as pd 


class TabularDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.y = y.to_numpy() if isinstance(y, pd.DataFrame) else y

    def __len__(self): # taking length of image_names list to see length of dataset
        return self.X.shape[0]

    def __getitem__(self, idx): # returns image, preprocessed image, and label
       return self.X[idx], self.y[idx]