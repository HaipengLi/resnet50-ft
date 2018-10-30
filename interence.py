import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from resnet50cifar import ResNet50_CIFAR
from dataloader import DataLoader


def inference(model, dataloader, cuda, limit=0):
    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using devide: {}".format(device))
        model.to(device)
    criterion = nn.CrossEntropyLoss()

    loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            if limit and i > limit:
                break

            inputs, labels = data
            if cuda:
                inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # loss
            loss += criterion(outputs, labels)

            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        loss = loss / len(dataloader)
        accuracy = correct / total

    return loss, accuracy


def inference_on_test(model, cuda, limit=0):
    dl = DataLoader()
    testloader = dl.load_test_set()
    print("Start inference")
    loss, accuracy = inference(model, testloader, cuda=cuda, limit=limit)
    print("Summary:")
    print("\tLoss: {}".format(loss))
    print("\tAccuracy: {}".format(accuracy))
    return loss, accuracy


def inference_on_train(model, cuda, limit=0):
    dl = DataLoader()
    trainloader = dl.load_train_set()
    print("Start inference")
    loss, accuracy = inference(model, trainloader, cuda=cuda, limit=limit)
    print("Summary:")
    print("\tLoss: {}".format(loss))
    print("\tAccuracy: {}".format(accuracy))
    return loss, accuracy


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--model', help='model file for ResNet50_CIFAR', required=True)
    arg_parser.add_argument('--cuda', action='store_true', default=False)
    args = arg_parser.parse_args()

    model = ResNet50_CIFAR()
    if args.cuda:
        model.load_state_dict(torch.load(args.model))
    else:
        model.load_state_dict(torch.load(args.model, map_location='cpu'))

    inference_on_train(model, args.cuda)
    inference_on_test(model, args.cuda)

