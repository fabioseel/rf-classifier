import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader, device: torch.device, max_num_batches: int = None):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()  # Set the model to training mode
    running_loss = 0.0

    epoch_correct = 0
    num_batches = len(train_loader)
    if max_num_batches is not None:
        num_batches = min(num_batches, max_num_batches)
    pbar = tqdm(enumerate(train_loader), total=num_batches)
    for i, (inputs, labels) in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass
        outputs = model(inputs)
        batch_correct =  num_correct(outputs, labels)
        epoch_correct += batch_correct
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pbar.set_postfix({'Acc.:': batch_correct/len(labels), 'Loss': loss.item()})
        if i == num_batches-1:
            avg_loss =running_loss/num_batches
            avg_acc = epoch_correct/min(num_batches*train_loader.batch_size, len(train_loader.dataset))
            pbar.set_postfix({'Acc.:': avg_acc, 'Loss': avg_loss})

    return avg_loss, avg_acc

def validate(model: nn.Module, dataloader: DataLoader, device: torch.device, abort_batch = None, verbose = True):
    # Evaluate the model on the test data
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        num_iter = abort_batch if abort_batch is not None else len(dataloader)

        all_outputs =[]
        all_labels = []
        pbar = tqdm(enumerate(dataloader), total=num_iter, disable=not verbose)
        for i, (inputs, labels) in pbar:
            all_labels.extend(labels)
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            all_outputs.extend(outputs.detach().cpu())
            total += labels.size(0)
            epoch_correct = num_correct(outputs, labels)
            correct += epoch_correct
            pbar.set_postfix({'Acc.:': epoch_correct/len(labels)})
            if abort_batch is not None:
                if i > abort_batch:
                    break
            if i == num_iter-1:
                accuracy = correct / total
                pbar.set_postfix({'Acc.:': accuracy})
    return accuracy, all_outputs, all_labels


def num_correct(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    return (predicted == labels).sum().item()