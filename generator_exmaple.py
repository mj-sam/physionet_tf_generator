"""
Load the individual numpy arrays into partition
"""
#include only right and left hand movement
n_classes = 2

L_data_train = glob.glob("./train/Left/S[0-9]*_[0-9]*.npy" , recursive=True)
R_data_train = glob.glob("./train/Right/S[0-9]*_[0-9]*.npy", recursive=True)

L_data_validation = glob.glob("./validation/Left/S[0-9]*_[0-9]*.npy" , recursive=True)
R_data_validation = glob.glob("./validation/Right/S[0-9]*_[0-9]*.npy", recursive=True)

L_data_test = glob.glob("./test/Left/S[0-9]*_[0-9]*.npy" , recursive=True)
R_data_test = glob.glob("./test/Right/S[0-9]*_[0-9]*.npy", recursive=True)


train_LR            =   [(L_data_train[i], tf.keras.utils.to_categorical(0,num_classes=n_classes)) for i in range(len(L_data_train))] + \
                        [(R_data_train[i], tf.keras.utils.to_categorical(1,num_classes=n_classes)) for i in range(len(R_data_train))]

validation_LR       =   [(L_data_validation[i], tf.keras.utils.to_categorical(0,num_classes=n_classes)) for i in range(len(L_data_validation))] + \
                        [(R_data_validation[i], tf.keras.utils.to_categorical(1,num_classes=n_classes)) for i in range(len(R_data_validation))]

test_LR             =   [(L_data_test[i], tf.keras.utils.to_categorical(0,num_classes=n_classes)) for i in range(len(L_data_test))] + \
                        [(R_data_test[i], tf.keras.utils.to_categorical(1,num_classes=n_classes)) for i in range(len(R_data_test))]

random.seed(n_classes)
random.shuffle(train_LR)
random.shuffle(validation_LR)
random.shuffle(test_LR)

partition_LR = {}

partition_LR['train']      = train_LR
partition_LR['validation'] = validation_LR
partition_LR['test']       = test_LR

# Parameters
params = {'dim': (22,560),
          'batch_size': 16,
          'n_classes': n_classes,
          'shuffle': True}

# Define the generators
training_generator_LR   = DataGenerator(partition_LR['train'], **params)
validation_generator_LR = DataGenerator(partition_LR['validation'], **params)
