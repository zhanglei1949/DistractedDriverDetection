# Parameters to specify for the run
batch_size = 144 # Try 48; 72; 109; 144; 218; 327 depending on the computing power
n_epoch = 20
learning_rate = 0.0001
n_layers_train = 8 # last n_layers_train layers will be trained
data_augmentation = 0 # 1 to augment image by transformation in the training phase
