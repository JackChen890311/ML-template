import torch.nn as nn

from constant import CONSTANT
from dataloader import MyDataloader

C = CONSTANT()


class MyModel(nn.Module):
    # Implement your model here
    def __init__(self):
        # Initialize your model object
        super(MyModel, self).__init__()
        pass

    def forward(self, x):
        # Return the output of model given the input x
        pass


if __name__ == '__main__':
    model = MyModel().to(C.device)
    print(model)
    
    dataloaders = MyDataloader()
    dataloaders.setup(['valid'])

    for x,y in dataloaders.loader['valid']:
        x = x.to(C.device)
        yhat = model(x)
        print(x.shape,y.shape)
        print(yhat.shape)
        break