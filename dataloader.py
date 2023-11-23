from torch.utils.data import DataLoader, Dataset

from constant import CONSTANT

C = CONSTANT()


class MyDataset(Dataset):
    # Implement your dataset here
    def __init__(self, data_path):
        # Initialize your dataset object
        super().__init__()
        pass

    def __len__(self):
        # Return the length of your dataset
        pass
    
    def __getitem__(self, idx):
        # Return an item pair, e.g. dataset[idx] and its label
        pass


class MyDataloader():
    def __init__(self):
        self.loader = {}

    def setup(self, types):
        print('Loading Data...')

        mapping = {
            'train':[C.data_path_train, True],
            'valid':[C.data_path_valid, False],
            'test' :[C.data_path_test, False],
                   }
        setupNames = list(set(types) & set(mapping.keys()))
        
        for name in setupNames:
            path, preprocess, shuffle = mapping[name]
            dataset = MyDataset(path, preprocess)
            self.loader[name] = self.loader_prepare(dataset, shuffle)
        
        if setupNames:
            print('Preparation Done! Use dataloader.loader[{type}] to access each loader.')
        else:
            print('Error: There is nothing to set up')

    def loader_prepare(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=C.bs,
            num_workers=C.nw,
            shuffle=shuffle,
            pin_memory=C.pm
        )


if __name__ == '__main__':
    dataloaders = MyDataloader()
    dataloaders.setup(['valid'])

    for x,y in dataloaders.loader['valid']:
        print(x.shape, y.shape)
        break