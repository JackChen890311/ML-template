import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from constant import CONSTANT


class Preprocess():
    def __init__(self):
        self.train_pre = T.Compose([
            T.Resize([256, 256]),
            T.RandomCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.test_pre = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class MyDataset(Dataset):
    # Implement your dataset here
    def __init__(self, data_path, preprocess):
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
        self.P = Preprocess()
        self.loader = {}

    def setup(self, types):
        print('Loading Data...')

        mapping = {
            'train':[self.C.data_path, self.P.train_pre, True],
            'valid':[self.C.data_path_valid, self.P.test_pre, False],
            'test' :[self.C.data_path_test, self.P.test_pre, False],
                   }
        setupNames = list(set(types) & set(mapping.keys()))
        
        for name in setupNames:
            path, preprocess, shuffle = mapping[name]
            self.loader[name] = self.loader_prepare(MyDataset(path, preprocess), shuffle)
        
        if setupNames:
            print('Preparation Done! Use dataloader.loader[{type}] to access each loader.')
        else:
            print('Error: There is nothing to set up')

    def loader_prepare(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.C.bs,
            num_workers=self.C.nw,
            shuffle=shuffle,
            pin_memory=self.C.pm
        )


if __name__ == '__main__':
    dataloaders = MyDataloader()
    dataloaders.setup(['test'])

    for x,y in dataloaders.loader['test']:
        print(x.shape, y.shape)
        break