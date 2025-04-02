import torch
import torch.nn as nn

from .constants import ACTIVATION_FUNCTION_MAPPING

class MLP(nn.Module):
   def __init__(self, hyperparameters:dict):
       super().__init__()

       self.model = self._create_model(hyperparameters)

   def forward(self, x:torch.Tensor):
       x = self.model(x)
       return x


   def _create_layer(self, layer_type:str, input_dim:int, output_dim:int):
       layer = None
       if layer_type == "linear":
           layer = nn.Linear(in_features=input_dim, out_features=output_dim)
       return layer
   
   def _create_fn(self, fn_type:str):
       if fn_type in ACTIVATION_FUNCTION_MAPPING:
           return ACTIVATION_FUNCTION_MAPPING[fn_type]
       return lambda x: x

   def _create_model(self, hyperparameters:dict):
       input_dim = hyperparameters["input_dim"]
       architecture = []
    

       for idx in range(hyperparameters["num_hidden_layers"]):
           output_dim = max(input_dim // 2, 1)

           architecture.append(self._create_layer(hyperparameters["hidden_layer_type"], input_dim, output_dim))
           architecture.append(self._create_fn(hyperparameters["hidden_activation_function"]))

           input_dim = max(input_dim // 2, 1)
       
       architecture.append(self._create_layer(hyperparameters["hidden_layer_type"], input_dim, 1))
       architecture.append(self._create_fn(hyperparameters["output_activation_function"]))

       model = nn.Sequential(*architecture)
       return model
