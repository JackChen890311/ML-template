from torch.utils.data import DataLoader, Dataset

from constant import CONSTANT

class MyDataset(Dataset):
    def __init__(self, data_path):
        # Implement your dataset here
        pass

    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass    


class MyDataloader():
    def __init__(self):
        super().__init__()
        self.C = CONSTANT()

    def setup_all(self):
        print('Loading Data...')
        self.train_dataset = MyDataset(self.C.data_path)
        self.train_loader = self.loader_prepare(self.train_dataset, True)
        del self.train_dataset

        self.valid_dataset = MyDataset(self.C.data_path_valid)
        self.valid_loader = self.loader_prepare(self.valid_dataset, True)
        del self.valid_dataset

        self.test_dataset = MyDataset(self.C.data_path_test) 
        self.test_loader = self.loader_prepare(self.test_dataset, True)
        del self.test_dataset
        print('Preparation Done!')
    
    def setup_test(self):
        print('Loading Data...')
        self.test_dataset = MyDataset(self.C.data_path_test) 
        self.test_loader = self.loader_prepare(self.test_dataset, True)
        del self.test_dataset
        print('Preparation Done!')

    def loader_prepare(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.C.bs,
            num_workers=self.C.nw,
            shuffle=shuffle,
            pin_memory=self.C.pm,
        )

if __name__ == '__main__':
    dataloaders = MyDataloader()
    dataloaders.setup_test()

    for x,y in dataloaders.test_loader:
        print(x.shape,len(y))
        break