import os

from os.path import join as path_join

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.dataset import create_datasets_from_json
from utils.utils import freeze_weights, unfreeze_weights


def train_clip(config):
    os.makedirs(config.PATH_TO_SAVE_MODEL_WEIGHTS, exist_ok=True)
    if config.PATH_TO_WEIGHTS is not None:
        config.MODEL.load_state_dict(torch.load(config.PATH_TO_WEIGHTS))
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.MODEL.to(DEVICE)
    criterion = config.CRITERION

    train_dataset, valid_dataset = create_datasets_from_json(
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
    for stage_name in config.TRAINING_STAGES:
        print(stage_name)

        parameters_of_stage = config.TRAINING_STAGES[stage_name]
        if 'freeze' in parameters_of_stage:
            freezed_parameters = parameters_of_stage['freeze']
            for name_params in freezed_parameters:
                freeze_weights(**freezed_parameters[name_params])
        if 'unfreeze' in parameters_of_stage:
            unfreezed_parameters = parameters_of_stage['unfreeze']
            for name_params in unfreezed_parameters:
                unfreeze_weights(**unfreezed_parameters[name_params])

        n_epoch = parameters_of_stage['n_epoch']

        optimizer_params = {key: parameters_of_stage[key] for key in parameters_of_stage if key in ['lr', 'params', 'base_optimizer']}
        optimizer = config.OPTIMIZER(**optimizer_params)
        is_sam_optimizer = True if 'base_optimizer' in optimizer_params else False

        for i in range(n_epoch):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, is_sam_optimizer)
            valid_loss = eval_epoch(model, valid_loader, criterion, DEVICE)
            if valid_loss < min_val_loss and valid_loss < train_loss:
                min_val_loss = valid_loss
                best_epoch = i + 1
                torch.save(
                    model.state_dict(),
                    path_join(config.PATH_TO_SAVE_MODEL_WEIGHTS, f'model_{stage_name}_{best_epoch}.pth')
                )
            print(f'Epoch {i + 1}/{n_epoch}\tTrain loss: {train_loss:.4f}; Valid image loss: {valid_loss:.4f}')
    print(f'Best epoch: {best_epoch}\t Valid loss: {min_val_loss}')


def train_epoch(model, dataloader, optimizer, criterion, device, is_sam_optimizer=False):
    model.train()
    train_loss = 0
    count = 0
    for batch in tqdm(dataloader, leave=False):
        image, text = batch['image'].to(device), batch['text'].to(device)
        batch_size_image, batch_size_text = image.size(0), text.size(0)

        labels_image, labels_text = torch.tensor([_ for _ in range(batch_size_image)]).to(device), torch.tensor([_ for _ in range(batch_size_text)]).to(device)
        if is_sam_optimizer:
            def closure():
                logits_image, logits_text = model((image, text))
                loss = criterion(logits_image, labels_image) + criterion(logits_text, labels_text)
                loss.backward()
                return loss

            logits_image, logits_text = model((image, text))
            loss = criterion(logits_image, labels_image) + criterion(logits_text, labels_text)
            loss.backward()
            optimizer.step(closure)
            optimizer.zero_grad()
            train_loss += loss.item()
        else:
            logits_image, logits_text = model((image, text))
            loss = criterion(logits_image, labels_image) + criterion(logits_text, labels_text)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        batch_size_image, batch_size_text = image.size(0), text.size(0)

        labels_image, labels_text = torch.tensor([_ for _ in range(batch_size_image)]).to(device), torch.tensor([_ for _ in range(batch_size_text)]).to(device)
        logits_image, logits_text = model((image, text))

        loss = criterion(logits_image, labels_image) + criterion(logits_text, labels_text)
        
        eval_loss += loss.item()
        count += 1

    return eval_loss / count
