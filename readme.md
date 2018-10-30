I implemented the second problem: "Run validation of the model every few training epochs on validation or test set of the dataset
and save the model with the best validation error."

Basically I refactored the orignal `imagenet_finetune.py`.

### List of files:
- `config.py`: contains the config of fine tuning
- `dataloader.py`: a wrapper class that allows easy cifar-10 data loading
- `imagenet_finetune.py`: contains the logic of fine tuning
- `inference.py`: contains convenient functions to do validation on train/test data set for given model
- `resnet50cifar.py`: contains the definition of the NN model

### Instructions:
To do fine tuning with validation on train/test set and saving models, just run
```
python imagenet_finetune.py
```
It will output logs about errors and save the models including the best one.
Finally it will output which epoch gives you the best model.


### Result:
I plot a graph showing loss & accuracy of different epochs.