from torch import device

class CONSTANT():
    def __init__(self):
        self.epochs = 100
        self.lr = 1e-3
        self.wd = 1e-5
        self.bs = 32
        self.nw = 16
        self.pm = True
        self.milestones = [30, 80]
        self.gamma = 0.5
        self.patience = 20
        self.verbose = 10
        self.device = device('cuda:0')

        self.data_path = 'data/training'
        self.data_path_valid = 'data/validation'
        self.data_path_test = 'data/testing'


        