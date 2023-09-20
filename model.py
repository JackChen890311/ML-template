import torch.nn as nn
import torch.nn.functional as F

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
    from constant import CONSTANT
    C = CONSTANT()
    model = MyModel()
    print(model)

    from dataloader import MyDataloader
    dataloaders = MyDataloader()
    dataloaders.setup(['test'])

    for x,y in dataloaders.test_loader:
        x = x.to(C.device)
        yhat = model(x)
        print(x.shape,y.shape)
        print(yhat.shape)
        break