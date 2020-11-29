from model import Unet, SegNet
from utils import get_data_paths
from sklearn.model_selection import train_test_split
from data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras import backend as K
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

def main():
    # 1936 x 1216
    input_size = (320, 480, 3)
    classes = 20
    train_dataset_x = '../seg_train_images/seg_train_images'
    train_dataset_y = '../seg_train_annotations/seg_train_annotations'
    test_size = 0.2
    batch_size = 8

    datasets_paths = get_data_paths(train_dataset_x, train_dataset_y)
    train_data, test_data = train_test_split(datasets_paths, test_size=test_size)
    net = Unet(input_size, classes)
    #net = SegNet(input_size, classes)
    net.summary()
    train_gen =  DataGenerator(train_data, input_size, classes, batch_size)
    val_gen =  DataGenerator(test_data, input_size, classes, batch_size)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=9, verbose=1),
        ModelCheckpoint('checkpoint/ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    ]

    net.compile(optimizer=Adam(1e-3),loss=categorical_crossentropy)
    history = net.fit_generator(
        train_gen, 
        steps_per_epoch=train_gen.num_batches_per_epoch,
        validation_data=val_gen,
        validation_steps=val_gen.num_batches_per_epoch,
        initial_epoch=0,
        epochs=50,
        callbacks=callbacks
    )
    net.save_weights('checkpoint/first_stage.h5')

    train_data, test_data = train_test_split(datasets_paths, test_size=test_size)
    train_gen =  DataGenerator(train_data, input_size, classes, batch_size)
    val_gen =  DataGenerator(test_data, input_size, classes, batch_size)

    net.compile(optimizer=Adam(1e-4),loss=categorical_crossentropy)
    history = net.fit_generator(
        train_gen, 
        steps_per_epoch=train_gen.num_batches_per_epoch,
        validation_data=val_gen,
        validation_steps=val_gen.num_batches_per_epoch,
        initial_epoch=50,
        epochs=100,
        callbacks=callbacks
    )
    net.save_weights('checkpoint/final_stage.h5')

if __name__ == "__main__":
    main()
