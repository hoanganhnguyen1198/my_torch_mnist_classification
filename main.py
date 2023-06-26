import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

import cnn_model

BATCH_SIZE = 100
LEARNING_RATE = 0.001
MOMENTUM = 0.9  # Read more: http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
EPOCHS = 5


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for i, data in enumerate(dataloader):
        # data is a list of [inputs, labels]
        X, y = data

        # forward-pass
        pred = model(X)

        # calculate loss
        loss = loss_fn(pred, y)

        # backward-pass
        loss.backward()

        # optimize
        optimizer.step()

        # zero out all the parameter gradients
        optimizer.zero_grad()

        if i % 100 == 0:
            loss, current = loss.item(), (i + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    train_set = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=ToTensor()
    )
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    test_set = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=ToTensor()
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


    model = cnn_model.CNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    for i in range(5):
        print(f"Epoch {i+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)
    print('Finished Training')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
            for X, y in test_loader:
                pred = model(X)
                pred = pred.argmax(1)
                for label, prediction in zip(y, pred):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
            
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    torch.save(model.state_dict(), "mnist_classifier_model.pth")
    print("Saved PyTorch Model State to mnist_classifier_model.pth")