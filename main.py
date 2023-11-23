import os
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import MyModel
from constant import CONSTANT
from dataloader import MyDataloader

C = CONSTANT()


def train(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0
    for idx, (x,y) in enumerate(loader):
        x,y = x.to(C.device),y.to(C.device)
        yhat = model(x)
        loss = loss_fn(yhat,y)

        total_loss += loss.item()
        loss /= C.accu_step
        loss.backward()

        if idx % C.accu_step == 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        # Add your own metrics here
    
    if len(loader) % C.accu_step != 0:
        optimizer.step()
        optimizer.zero_grad()
    return total_loss/len(loader)


def valid(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(C.device),y.to(C.device)
            yhat = model(x)
            loss = loss_fn(yhat,y)
            total_loss += loss.item()
            # Add your own metrics here
    return total_loss/len(loader)


def test(model, loader):
    model.eval()
    result = []
    with torch.no_grad():
        for x in loader:
            x = x.to(C.device)
            yhat = model(x)
            result.append(yhat)
            # Add your own metrics here
    return result


def run_inference(modelPath):
    # Model
    model = MyModel()
    state = model.load_state_dict(torch.load(modelPath))
    assert len(state.unexpected_keys) == 0 and len(state.missing_keys) == 0
    print('Success! Model loaded from %s'%modelPath)

    model = model.to(C.device)
    model.eval()

    # Data
    dataloaders = MyDataloader()
    dataloaders.setup(['test'])

    # Inference
    result = test(model, dataloaders.loader['test'])
    return result


def myplot(config):
    plt.title(config['title'])
    plt.xlabel(config['xlabel'])
    plt.ylabel(config['ylabel'])
    for label in config['data']:
        plt.plot(config['data'][label][0], config['data'][label][1], label=label)
    plt.legend()
    plt.savefig(config['savefig'])
    plt.clf()


def message_handler(msg, dir_name):
    with open(os.path.join('output', dir_name, 'log.log'), 'a') as f:
        f.write(msg+'\n')
    print(msg)


def main():
    # Load model and data
    model = MyModel()
    model = model.to(C.device)
    dataloaders = MyDataloader()
    dataloaders.setup(['train', 'valid'])

    # You can adjust these as your need
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloaders.loader['train'])*C.epochs)

    # Set up output directory
    start_time = time.time()
    start_time_str = str(time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()))
    os.makedirs('output/%s'%start_time_str, exist_ok=True)

    begin_msg = '==================== Parameter Information ===================='
    const_msg = [f"{key}: {value}" for key, value in vars(C).items()]
    train_msg = [
        'Optimizer:'+str(optimizer),
        'Scheduler:'+str(scheduler),
        'Loss Function:'+str(loss_fn),
        '\n==================== Model Structure ====================',
        str(model),
        '==================== Begin Training... ===================='
    ]
    for msg in begin_msg + const_msg + train_msg:
        message_handler(msg, start_time_str)
    
    # Start training
    train_losses = []
    valid_losses = []
    p_cnt = 0
    best_valid_loss = 1e10

    for e in tqdm(range(1,1+C.epochs)):
        # Train and valid step
        train_loss = train(model, dataloaders.loader['train'], optimizer, scheduler, loss_fn)
        valid_loss = valid(model, dataloaders.loader['valid'], loss_fn)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # Plot loss and print training information
        if e % C.verbose == 0:
            epoch_msg = f'Epoch = {e}, Train / Valid Loss = {round(train_loss, 6)} / {round(valid_loss, 6)}'
            message_handler(epoch_msg, start_time_str)
            
            config = {
                'title':'Loss Plot',
                'xlabel':'Epochs',
                'ylabel':'Loss',
                'data':{
                    'Train':[list(range(1,1+e)), train_losses],
                    'Valid':[list(range(1,1+e)), valid_losses]
                },
                'savefig':f'output/{start_time_str}/loss.png'
            }
            myplot(config)

        # Save best model and early stopping
        if valid_loss < best_valid_loss:
            p_cnt = 0
            best_valid_loss = valid_loss
            save_msg = f'Save model on epoch {e}. Best valid loss: {round(best_valid_loss, 6)}.'
            message_handler(save_msg, start_time_str)
            torch.save(model.state_dict(), 'output/%s/model.pt'%start_time_str)
        else:
            p_cnt += 1
            if p_cnt == C.patience:
                stop_msg = f'Early Stopping at epoch {e}.'
                message_handler(stop_msg, start_time_str)

    
    # End of training
    end_time = time.time()
    end_time_str = str(time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()))
    ending_msg = f'Ending at epoch {e}. Best valid loss: {round(best_valid_loss, 6)}.'
    message_handler(ending_msg, start_time_str)
    time_msg = f'Start time: {start_time_str}, End time: {end_time_str}, Training takes {round((end_time-start_time)/3600, 2)} hours.'
    message_handler(time_msg, start_time_str)
    


if __name__ == '__main__':
    main()

