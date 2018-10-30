import torch
from resnet50cifar import ResNet50_CIFAR
import interence


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
        model.load_state_dict(torch.load(model_name))
        train_loss, train_accuracy = interence.inference_on_train(model, cuda=False)
        test_loss, test_accuracy = interence.inference_on_test(model, cuda=False)
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)

    with open("result.txt", 'w') as f:
        f.write(str(model_list))
        f.write(str(train_loss_list))
        f.write(str(train_accuracy_list))
        f.write(str(test_loss_list))
        f.write(str(test_accuracy_list))
