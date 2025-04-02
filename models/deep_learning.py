import torch
from torch.utils.data import DataLoader

class DeepLearning():
    def __init__(self, hyperparameters:dict):
        self.hyperparameters = hyperparameters

    def train(self, X, y):
        raise NotImplementedError("Subclass of Model should implement train")
    
    def __call__(self, X):
        raise NotImplementedError("Subclass of Model should implement __call__")

class MLPTrainer(DeepLearning):
    from .architecture import MLP
    from .constants import LOSS_FUNCTIONS
    from datasets.dataset import TabularDataset
    def __init__(self, hyperparameters: dict, device = "cuda" if torch.cuda.is_available() else "cpu", seed: int = 1337):
        super().__init__(hyperparameters)
        torch.manual_seed(seed)
        self.device = device
        self.model = self.MLP(self.hyperparameters) # passes the key values. when passing in fit_passing it'll read as fit_passing = True, instead of us writing out 
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = hyperparameters["learning_rate"])
        self.loss_fn = self.LOSS_FUNCTIONS[hyperparameters["loss_fn"]]


    def train(self, X, y):  
            dataset = self.TabularDataset(X,y)
            train_dataloader = DataLoader(dataset, batch_size = self.hyperparameters["batch_size"])
             

            train_loss = 0
            train_steps = 0
            self.model.train()
            # Train loop
            for epoch in range(self.hyperparameters["num_epochs"]):
                for idx, batch_data in enumerate(train_dataloader):
                    features, targets = batch_data
                    #  advantage of pytorch: only takes up memory bc we put them in gpu.
                    features, targets = features.to(self.device), targets.to(self.device) # moves to specific place in hardware.
                    features, targets = features.to(torch.float32), targets.to(torch.float32)
                    

                    self.optimizer.zero_grad() # sets gradient to 0 every time 

                    outputs = self.model(features) # apply model to images and get output 

                    loss = self.loss_fn(outputs, targets) # compute loss function
                    loss.backward() # partial derivative to find direction and strength that maximizes or minimizes the function 

                    train_loss += loss.item() # accumulating the losses (for logging)
                    train_steps += 1


                    self.optimizer.step() # using parameters to find how big jump should be 

            return train_loss / train_steps

    def __call__(self, X):
        output = self.model(X)
        return output 
    
    def reset(self):
        def weights_init(m):
            try:
                torch.nn.init.xavier_uniform_(m.weight.data)
            except AttributeError as e:
                pass
        self.model.apply(weights_init)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.hyperparameters["learning_rate"])