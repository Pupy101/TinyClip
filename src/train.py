import os

from os.path import join as path_join

import torch

from tqdm import tqdm

from .configurator import Configurator


def train_clip(config):
    os.makedirs(config.PATH_TO_SAVE_MODEL_WEIGHTS, exist_ok=True)
    configurator_training = Configurator(config)
    parameters = configurator_training.init_all()
    model = parameters['model']
    optimizer = parameters['optimizer']
    scheduler = parameters['scheduler']
    criterion = parameters['criterion']
    loaders = parameters['loaders']
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    min_val_loss = float('inf')
    best_epoch = 0    
    n_epoch = config.NUM_EPOCH
    accumulation = config.ACCUMULATE

    for i in range(n_epoch):
        train_loss = train_epoch(
            model, loaders['train'], optimizer, scheduler,
            criterion, DEVICE, accumulation
            )
        valid_loss = eval_epoch(model, loaders['valid'], criterion, DEVICE)
        if valid_loss < min_val_loss and valid_loss < train_loss:
            min_val_loss = valid_loss
            best_epoch = i + 1
            torch.save(
                model.state_dict(),
                path_join(
                    config.PATH_TO_SAVE_MODEL_WEIGHTS,
                    f'{best_epoch}.pth'
                )
            )
        print(f'Epoch {i + 1}/{n_epoch}\tTrain loss: {train_loss:.4f}; Valid image loss: {valid_loss:.4f}')
    print(f'Best epoch: {best_epoch}\t Valid loss: {min_val_loss}')


def train_epoch(
    model, dataloader, optimizer, scheduler, criterion, device, accumulation=False
):
    model.train()
    train_loss = 0
    count = 0
    for batch in tqdm(dataloader, leave=False):
        image, text = batch['image'].to(device), batch['text'].to(device)
        batch_size_image = image.size(0)

        labels_image = torch.tensor([_ for _ in range(batch_size_image)]).to(device),

        logits_image, _ = model((image, text))
        loss = criterion(logits_image, labels_image)
        # accumulating
        if (accumulation and count % 2) or not accumulation:
            optimizer.zero_grad()
        loss.backward()
        # accumulating
        if (accumulation and count % 2) or not accumulation:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        train_loss += loss.item()
        count += 1

    return train_loss / count

@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    eval_loss = 0
    count = 0
    for batch in tqdm(dataloader, leave=False):
        image, text = batch['image'].to(device), batch['text'].to(device)
        batch_size_image = image.size(0)

        labels_image = torch.tensor([_ for _ in range(batch_size_image)]).to(device)
        logits_image, _ = model((image, text))

        loss = criterion(logits_image, labels_image)
        
        eval_loss += loss.item()
        count += 1

    return eval_loss / count
