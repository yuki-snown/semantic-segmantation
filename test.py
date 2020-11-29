from model import Unet
from utils import get_data_paths
from sklearn.model_selection import train_test_split
from data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras import backend as K
import tensorflow as tf
from PIL import Image
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

def main():
    input_size = (320, 480, 3)
    classes = 21
    weight_path = './checkpoint/final_stage.h5'
    filename = '../seg_train_images/seg_train_images/train_1118.jpg'
    color_index = [
        [0,0,255],[193,214,0],[180,0,129],[255,121,166],
        [255,0,0],[65,166,1],[208,149,1],[255,255,0],
        [255,134,0],[0,152,225],[0,203,151],[85,255,50],
        [92,136,125],[69,47,142],[136,45,66],[0,255,255],
        [215,0,255],[180,131,135],[81,99,0],[86,62,67]
    ]
    net = Unet(input_size, classes)
    net.load_weights(weight_path)
    img = Image.open(filename).resize((input_size[1],input_size[0]))
    img = np.asarray(img, dtype='float32')
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = net.predict(img)
    b, w, h, c = pred.shape
    res = []
    for i in range(w):
        tmp = []
        for j in range(h):
            tmp.append(color_index[np.argmax(pred[0,i,j,:])])
        res.append(tmp)
    img = Image.fromarray(np.asarray(res, dtype='uint8'))
    img.save('result.png')

if __name__ == "__main__":
    main()