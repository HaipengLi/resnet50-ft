import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from resnet50cifar import ResNet50_CIFAR
import interence
from tqdm import tqdm
import config


def train(cuda=False, num_epoch=10, save_file=''):
    ## Define the training dataloader
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10('./data', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

    print("{} iterations each epoch".format(len(trainloader)))

    ## Create model, objective function and optimizer
    model = ResNet50_CIFAR()
    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using devide: {}".format(device))
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()),
                           lr=0.001, momentum=0.9)

    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    ## Do the training
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        print("Training {}/{} epoch".format(epoch + 1, num_epoch))
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, 0)):
            # get the inputs
            inputs, labels = data
            if cuda:
                inputs, labels = inputs.to(device), labels.to(device)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            # if i % 20 == 19:    # print every 20 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 20))
            #     running_loss = 0.0

        print("testing on training set..")
        train_loss, train_accuracy = interence.inference_on_train(model, cuda=True, limit=1000)
        print("testing on testing set..")
        test_loss, test_accuracy = interence.inference_on_test(model, cuda=True, limit=1000)

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)

        if save_file:
            torch.save(model.state_dict(), save_file.format(epoch + 1))
    print('Finished Training')
    # find the best model
    loss_np = np.add(train_loss_list, test_loss_list)
    best_idx = np.argmin(loss_np)
    print("Best model is in epoch {}".format(best_idx + 1))

    plt.figure()
    plt.plot(list(range(1, num_epoch + 1)), train_loss_list, label='train')
    plt.plot(list(range(1, num_epoch + 1)), test_loss_list, label='test')

    plt.ylabel('Loss')
    plt.title('Loss of different epochs')
    plt.xlabel('Epoch')
    plt.show()

    plt.figure()
    plt.plot(list(range(1, num_epoch + 1)), train_accuracy_list, label='train')
    plt.plot(list(range(1, num_epoch + 1)), test_accuracy_list, label='test')

    plt.ylabel('Accuracy')
    plt.title('Accuracy of different epochs')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == '__main__':
    train(cuda=config.CUDA, num_epoch=config.NUM_EPOCH, save_file=config.SAVE_FILE)
