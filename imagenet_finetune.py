import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from resnet50cifar import ResNet50_CIFAR
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

    train_accuracy = []
    test_accuracy = []
    train_loss = []
    test_loss = []

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
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        # TODO: using `inference.py`, print (accuracy, loss) for train and test set
        print("Loss")
        print("Loss for whole")
        if save_file:
            torch.save(model.state_dict(), save_file.format(epoch + 1))
    print('Finished Training')

    # load
    # the_model = TheModelClass(*args, **kwargs)
    # the_model.load_state_dict(torch.load(PATH))


if __name__ == '__main__':
    train(cuda=config.CUDA, num_epoch=config.NUM_EPOCH, save_file=config.SAVE_FILE)
