import torch
from resnet50cifar import ResNet50_CIFAR
import interence
import matplotlib.pyplot as plt


# TODO: for each model, get loss / accuracy for train / test set
# TODO: plot accuracy & loss
if __name__ == '__main__':
    id_range = range(1, 12)
    model_name_scheme = 'models/trained_epoch_{:02}.model'
    model_list = []
    for i in id_range:
        model_list.append(model_name_scheme.format(i))
    print(model_list)

    model = ResNet50_CIFAR()

    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    for model_name in model_list:
        print("testing on model: {}".format(model_name))
        model.load_state_dict(torch.load(model_name))
        train_loss, train_accuracy = interence.inference_on_train(model, cuda=True, limit=20)
        print("\ttrain_loss: {}".format(train_loss))
        print("\ttrain_accuracy: {}".format(train_accuracy))
        test_loss, test_accuracy = interence.inference_on_test(model, cuda=True, limit=20)
        print("\ttest_loss: {}".format(test_loss))
        print("\ttest_accuracy: {}".format(test_accuracy))

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)

    with open("result.txt", 'w') as f:
        f.writelines(str(model_list))
        f.writelines(str(train_loss_list))
        f.writelines(str(test_loss_list))
        f.writelines(str(train_accuracy_list))
        f.writelines(str(test_accuracy_list))

    plt.figure()
    plt.plot(list(id_range), train_loss_list, label='train')
    plt.plot(list(id_range), test_loss_list, label='test')

    plt.ylabel('Loss')
    plt.title('Loss of different epochs')
    plt.xlabel('Epoch')
    plt.show()

    plt.figure()
    plt.plot(list(id_range), train_accuracy_list, label='train')
    plt.plot(list(id_range), test_accuracy_list, label='test')

    plt.ylabel('Accuracy')
    plt.title('Accuracy of different epochs')
    plt.xlabel('Epoch')
    plt.show()
