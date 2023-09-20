import os
import time
import torch
import pickle as pk
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import MyModel
from dataloader import MyDataloader
from constant import CONSTANT

C = CONSTANT()

def train(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for x,y in train_loader:
        x = x.to(C.device)
        optimizer.zero_grad()
        result = model(x)
        loss = loss_fn(y,result)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(train_loader)


def test(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    for x,y in test_loader:
        x = x.to(C.device)
        result = model(x)
        loss = loss_fn(y,result)
        total_loss += loss.item()
    return total_loss/len(test_loader)


def main():
    model = MyModel()
    model = model.to(C.device)
    dataloaders = MyDataloader()
    dataloaders.setup(['train', 'valid', 'test'])
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=C.milestones, gamma=C.gamma)

    start_time = str(time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()))
    if not os.path.exists('output'):
        os.mkdir('output')
    os.mkdir('output/%s'%start_time)

    with open('output/%s/pata.txt'%start_time, 'w') as f:
        for key, value in vars(C).items():
            f.write(f"{key}: {value}\n")

    train_losses = []
    valid_losses = []
    p_cnt = 0
    best_valid_loss = 1e10

    for e in tqdm(range(1,1+C.epochs)):
        train_loss = train(model, dataloaders.train_loader, optimizer, loss_fn)
        valid_loss = test(model, dataloaders.valid_loader, loss_fn)
        scheduler.step()
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f'Epoch = {e}, Train / Valid Loss = {round(train_loss, 6)} / {round(valid_loss, 6)}')
       
        if valid_loss < best_valid_loss:
            p_cnt = 0
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'output/%s/model.pt'%start_time)
        else:
            p_cnt += 1
            if p_cnt == C.patience:
                print('Early Stopping at epoch', e)
                break
        
        if e % C.verbose == 0:
            print('Plotting Loss at epoch', e)
            x_axis = list(range(e))
            plt.plot(x_axis, train_losses, label='Train')
            plt.plot(x_axis, valid_losses, label='Valid')
            plt.legend()
            plt.savefig('output/%s/loss.png'%start_time)
            plt.clf()

            plt.plot(x_axis[-100:], train_losses[-100:], label='Train')
            plt.plot(x_axis[-100:], valid_losses[-100:], label='Valid')
            plt.legend()
            plt.savefig('output/%s/loss_last100.png'%start_time)
            plt.clf()

        with open('output/%s/losses.pickle'%start_time, 'wb') as file:
            pk.dump([train_losses, valid_losses, best_valid_loss], file)
        
    print(f'Ending at epoch {e}. Best valid loss: {round(best_valid_loss, 6)}')
    with open('output/%s/pata.txt'%start_time, 'a') as f:
        f.write(f"Ending at epoch {e}. Best valid loss: {round(best_valid_loss, 6)}\n")
        f.write('===== Model Structure =====')
        f.write('\n'+str(model)+'\n')

def test_model(path):
    model = MyModel()
    model.load_state_dict(torch.load(path))
    model = model.to(C.device)
    model.eval()
    dataloaders = MyDataloader()
    dataloaders.setup(['test'])
    loss_fn = torch.nn.MSELoss()
    loss = test(model, dataloaders.test_loader, loss_fn)
    return loss


if __name__ == '__main__':
    main()

