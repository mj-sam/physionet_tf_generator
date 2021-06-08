import tensorflow as tf

class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, list_examples, batch_size=64, dim=(64, 560),
                n_classes=2, shuffle=True):
        # Constructor of the data generator.
        self.dim = dim
        self.batch_size = batch_size
        self.list_examples = list_examples
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_examples) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_examples[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # This function is called at the end of each epoch.
        self.indexes = np.arange(len(self.list_examples))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Load individual numpy arrays and aggregate them to a batch.    
        X = np.empty([self.batch_size, self.dim[0], self.dim[1]], dtype=np.float32)
        # y is a one-hot encoded vector.
        y = np.empty([self.batch_size,self.n_classes], dtype=np.int16)
        # Generate data.
        for i, ID in enumerate(list_IDs_temp):
            # Load sample
            #X[i,:, :] = np.load(ID[0])[channels_eq.iloc[:]['physionet'],0:560]
            X[i,:, :] = np.load(ID[0])[:,0:560]
            # Load labels      
            y[i, :] = ID[1]
        return X, y