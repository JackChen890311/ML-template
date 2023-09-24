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
        optimizer.zero_grad()
        x,y = x.to(C.device),y.to(C.device)
        yhat = model(x)
        loss = loss_fn(yhat,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Add your own metrics here
    return total_loss/len(train_loader)


def test(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    for x,y in test_loader:
        x,y = x.to(C.device),y.to(C.device)
        yhat = model(x)
        loss = loss_fn(yhat,y)
        total_loss += loss.item()
        # Add your own metrics here
    return total_loss/len(test_loader)


def myplot(config):
    plt.title(config['title'])
    plt.xlabel(config['xlabel'])
    plt.ylabel(config['ylabel'])
    for label in config['data']:
        plt.plot(config['data'][label][0], config['data'][label][1], label=label)
    plt.legend()
    plt.savefig(config['savefig'])
    plt.clf()


def test_model(modelPath):
    # Model
    model = MyModel()
    model.load_state_dict(torch.load(modelPath))
    model = model.to(C.device)
    model.eval()

    # Data
    dataloaders = MyDataloader()
    dataloaders.setup(['test'])

    # Add your own metrics here
    loss_fn = torch.nn.MSELoss()
    loss = test(model, dataloaders.test_loader, loss_fn)
    return loss


def main():
    # Load model and data
    model = MyModel()
    model = model.to(C.device)
    dataloaders = MyDataloader()
    dataloaders.setup(['train', 'valid', 'test'])

    # You can adjust these as your need
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=C.milestones, gamma=C.gamma)

    # Set up output directory
    start_time = str(time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()))
    if not os.path.exists('output'):
        os.mkdir('output')
    os.mkdir('output/%s'%start_time)
    with open('output/%s/log.txt'%start_time, 'w') as f:
        for key, value in vars(C).items():
            f.write(f"{key}: {value}\n")
        f.write('Optimizer:'+str(optimizer)+'\n')
        f.write('Scheduler:'+str(scheduler)+'\n')
        f.write('\n===== Model Structure =====')
        f.write('\n'+str(model)+'\n')
        f.write('===== Begin Training... =====\n')

    # Start training
    train_losses = []
    valid_losses = []
    p_cnt = 0
    best_valid_loss = 1e10

    for e in tqdm(range(1,1+C.epochs)):
        # Train and valid step
        train_loss = train(model, dataloaders.loader['train'], optimizer, loss_fn)
        valid_loss = test(model, dataloaders.loader['test'], loss_fn)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f'Epoch = {e}, Train / Valid Loss = {round(train_loss, 6)} / {round(valid_loss, 6)}')
        scheduler.step()
        
        # Plot loss
        if e % C.verbose == 0:
            print('Plotting Loss at epoch', e)
            x_axis = list(range(e))
            config = {
                'title':'Loss',
                'xlabel':'Epochs',
                'ylabel':'Loss',
                'data':{
                    'Train':[x_axis, train_losses],
                    'Valid':[x_axis, valid_losses]
                },
                'savefig':'output/%s/loss.png'%start_time
            }
            myplot(config)

            config['data']['Train'] = [x_axis[-100:], train_losses[-100:]]
            config['data']['Valid'] = [x_axis[-100:], valid_losses[-100:]]
            config['savefig'] = 'output/%s/loss_last100.png'%start_time
            myplot(config)

        # Save best model and early stopping
        if valid_loss < best_valid_loss:
            p_cnt = 0
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'output/%s/model.pt'%start_time)
        else:
            p_cnt += 1
            if p_cnt == C.patience:
                print('Early Stopping at epoch', e)
                break
        
        # Write log
        with open('output/%s/log.txt'%start_time, 'a') as f:
            f.write(f"Epoch {e}: Best valid loss: {round(best_valid_loss, 6)}\n")
        with open('output/%s/losses.pickle'%start_time, 'wb') as file:
            pk.dump([train_losses, valid_losses, best_valid_loss], file)
    
    # Ends training
    print(f'Ending at epoch {e}. Best valid loss: {round(best_valid_loss, 6)}')
    with open('output/%s/log.txt'%start_time, 'a') as f:
        f.write(f"Ending at epoch {e}. Best valid loss: {round(best_valid_loss, 6)}\n")


if __name__ == '__main__':
    main()

