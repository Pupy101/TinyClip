import torch


def train_clip(config):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.MODEL.to(DEVICE)
    optimizer = config.OPTIMIZER(
        **config.OPTIMIZER_PARAMS
    )
    criterion = config.CRITERION


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    count = 0
    for batch in dataloader:
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
    for batch in dataloader:
        image, text = batch['image'].to(device), batch['text'].to(device)
        batch_size = image.size(0)
        labels = torch.tensor([_ for _ in range(1, batch_size + 1)]).to(device)
        with torch.no_grad():
            output = model((image, text))
            loss = criterion(output, labels)

        eval_loss += loss.item()
        count += 1

    return eval_loss / count

