from keras.utils import Sequence
from PIL import Image
import numpy as np
from utils import to_categorical_tensor, color_mapping

class DataGenerator(Sequence):
    def __init__(self, data, input_shape, classes, batch_size=1):
        self.data = data
        self.length = len(data)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.classes = classes
        self.num_batches_per_epoch = int((self.length - 1) / batch_size) + 1

    def __getitem__(self, idx):
        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > self.length:
            end_pos = self.length

        items = self.data[start_pos : end_pos]

        X, y= [], []

        for x_path, y_path in items:
            img = Image.open(x_path).resize((self.input_shape[1],self.input_shape[0]))
            img = np.asarray(img)
            img = img / 255.0
            X.append(img.tolist())
            img = Image.open(y_path).resize((self.input_shape[1], self.input_shape[0]))
            img = np.asarray(img)
            img = color_mapping(img)
            y.append(img.tolist())

        X = np.asarray(X, dtype='float32')
        y = np.asarray(y, dtype='float32')
 
        return X, to_categorical_tensor(y, self.classes)

    def __len__(self):
        return self.num_batches_per_epoch