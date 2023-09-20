from torch.utils.data import DataLoader, Dataset

from constant import CONSTANT

class MyDataset(Dataset):
    # Implement your dataset here
    def __init__(self, data_path):
        # Initialize your dataset object
        pass

    def __len__(self):
        # Return the length of your dataset
        pass
    
    def __getitem__(self, idx):
        # Return an item pair, e.g. dataset[idx] and its label
        pass    


class MyDataloader():
    def __init__(self):
        super().__init__()
        self.C = CONSTANT()

    def setup(self, types):
        flag = False
        print('Loading Data...')
        if 'train' in types:
            flag = True
            self.train_dataset = MyDataset(self.C.data_path)
            self.train_loader = self.loader_prepare(self.train_dataset, True)
            del self.train_dataset

        if 'valid' in types:
            flag = True
            self.valid_dataset = MyDataset(self.C.data_path_valid)
            self.valid_loader = self.loader_prepare(self.valid_dataset, True)
            del self.valid_dataset

        if 'test' in types:
            flag = True
            self.test_dataset = MyDataset(self.C.data_path_test) 
            self.test_loader = self.loader_prepare(self.test_dataset, True)
            del self.test_dataset
        
        if flag:
            print('Preparation Done! Use dataloader.{type}_loader to access each loader.')
        else:
            print('Error: There is nothing to set up')

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
    dataloaders.setup(['test'])

    for x,y in dataloaders.test_loader:
        print(x.shape, y.shape)
        break