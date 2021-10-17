from os.path import join as path_join

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.dataset import create_dataset


def train_clip(config):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.MODEL.to(DEVICE)
    optimizer = config.OPTIMIZER(
        **config.OPTIMIZER_PARAMS
    )
    criterion = config.CRITERION

    train_dataset, valid_dataset = create_dataset(
        **config.DATASET_PARAMS
    )

    train_loader = DataLoader(
        train_dataset,
        **config.LOADER_PARAMS['train']
    )
    valid_loader = DataLoader(
        valid_dataset,
        **config.LOADER_PARAMS['valid']
    )
    min_val_loss = float('inf')
    best_epoch = 0
    for i in range(config.N_EPOCH):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        valid_loss = eval_epoch(model, valid_loader, optimizer, criterion, DEVICE)
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
            best_epoch = i + 1
            torch.save(
                model.state_dict(),
                path_join(config.PATH_TO_SAVE_MODEL_WEIGHTS, f'model_{best_epoch}.pth')
            )
        print(f'Epoch {i + 1}/{config.N_EPOCH + 1}\t Train loss: {train_loss}, Valid loss: {valid_loss}')
    print(f'Best epoch: {best_epoch}\t Valid loss: {min_val_loss}')


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    count = 0
    for batch in tqdm(dataloader):
        image, text = batch['image'].to(device), batch['text'].to(device)
        batch_size = image.size(0)
        output = model((image, text))
        labels = torch.tensor([_ for _ in range(1, batch_size + 1)]).to(device)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        count += 1

    return train_loss / count


def eval_epoch(model, dataloader, optimizer, criterion, device):
    model.eval()
    eval_loss = 0
    count = 0
    for batch in tqdm(dataloader):
        image, text = batch['image'].to(device), batch['text'].to(device)
        batch_size = image.size(0)
        labels = torch.tensor([_ for _ in range(1, batch_size + 1)]).to(device)
        with torch.no_grad():
            output = model((image, text))
            loss = criterion(output, labels)

        eval_loss += loss.item()
        count += 1

    return eval_loss / count
