import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class DataLoader:
    def __init__(self):
        self.transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))])
        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.trainset = None
        self.testset = None

    def load_train_set(self):
        if self.trainset is None:
            self.trainset = datasets.CIFAR10('./data/train', train=True, download=True, transform=self.transform)

        return torch.utils.data.DataLoader(self.trainset, batch_size=4, shuffle=True, num_workers=2)

    def load_test_set(self):
        if self.testset is None:
            self.testset = datasets.CIFAR10('./data/test', train=False, download=True, transform=self.transform)

        return torch.utils.data.DataLoader(self.testset, batch_size=4, shuffle=False, num_workers=2)

    def imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def show_sample_img(self, class_name):
        if class_name not in self.classes:
            raise ValueError("Unknown class: {}".format(class_name))
        trainloader = self.load_train_set()
        for data in trainloader:
            inputs, labels = data
            labels = map(lambda x: self.classes[x], labels)
            classes = []
            for label in labels:
                classes.append(label)
            if class_name not in classes:
                continue

            idx = classes.index(class_name)
            img = inputs[idx]
            self.imshow(img)
            break


if __name__ == '__main__':
    dl = DataLoader()
    for class_name in dl.classes:
        dl.show_sample_img(class_name)
    print("Labels: {}".format(dl.classes))
