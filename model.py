import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Implement your model here
        pass

    def forward(self, x):
        pass


if __name__ == '__main__':
    from constant import CONSTANT
    C = CONSTANT()
    model = MyModel()
    print(model)