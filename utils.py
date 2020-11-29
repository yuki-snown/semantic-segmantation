import os
import random
import numpy as np
from keras.utils import to_categorical

def to_categorical_tensor( x3d, n_cls ) :
      batch_size, n_rows, n_cols = x3d.shape
      x1d = x3d.ravel()
      y1d = to_categorical( x1d, num_classes = n_cls )
      y4d = y1d.reshape( [ batch_size, n_rows, n_cols, n_cls ] )
      return y4d

def get_data_paths(folder_x, folder_y, random_seed=10101):
    paths = []
    for file_ in os.listdir(folder_x):
        file_name = file_.split('.')[0]
        X = '{}/{}.jpg'.format(folder_x, file_name)
        y = '{}/{}.png'.format(folder_y, file_name)
        paths.append([X,y])
    random.seed(random_seed)
    random.shuffle(paths)
    return paths

def get_color_index(rgb):
    color_index = [
        [0,0,255],[193,214,0],[180,0,129],[255,121,166],
        [255,0,0],[65,166,1],[208,149,1],[255,255,0],
        [255,134,0],[0,152,225],[0,203,151],[85,255,50],
        [92,136,125],[69,47,142],[136,45,66],[0,255,255],
        [215,0,255],[180,131,135],[81,99,0],[86,62,67]
    ]
    for ind, obj in enumerate(color_index):
        if rgb == obj: return ind

def color_mapping(img):
    color_map = []
    w, h = img.shape[:2]
    for i in range(w):
        tmp = []
        for j in range(h):
            tmp.append(get_color_index(img[i,j,:].tolist()))
        color_map.append(tmp)
    return np.asarray(color_map)