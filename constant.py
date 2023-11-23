import torch


class CONSTANT():
    def __init__(self):
        self.epochs = 50
        self.lr = 1e-5
        self.wd = 1e-6
        self.bs = 32
        self.nw = 4
        self.pm = True
        self.accu_step = 1
        self.patience = 10
        self.verbose = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_path_train = 'data/train'
        self.data_path_valid = 'data/valid'
        self.data_path_test = 'data/test'


        