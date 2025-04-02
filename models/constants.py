import torch

ACTIVATION_FUNCTION_MAPPING = {
    "relu": torch.nn.ReLU(),
    "tanh": torch.nn.Tanh(),
    "sigmoid": torch.nn.Sigmoid(),
    "none": lambda x: x
}


LOSS_FUNCTIONS = {
    "mse": torch.nn.MSELoss(),
    "mae": torch.nn.L1Loss(),
    "bce": torch.nn.BCELoss()
}

