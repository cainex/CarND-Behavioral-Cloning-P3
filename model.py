import csv
import cv2

from keras import backend as K

if 'tensorflow' == K.backend():
    print('setting Tensorflow session configuration')
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import argparse

def load_data(path):
    lines = []
    with open('{}/driving_log.csv'.format(path)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    lines.pop(0)

    return lines

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                left_name = batch_sample[1]
                right_name = batch_sample[2]

                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)

                left_image = mpimg.imread(left_name)
                left_angle = float(batch_sample[3]) + correction
                images.append(left_image)
                angles.append(left_angle)
                images.append(np.fliplr(left_image))
                angles.append(-left_angle)

                right_image = mpimg.imread(right_name)
                right_angle = float(batch_sample[3]) - correction
                images.append(right_image)
                angles.append(right_angle)
                images.append(np.fliplr(right_image))
                angles.append(-right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def build_model():
    """
    This will build the Keras model. This follows the NVIDIA self-driving car neural network
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Self-driving car simulation training script')
    parser.add_argument('-p', '--path', help='path to dataset CSV', dest='csv_path', type=str)
    parser.add_argument('-b', '--batch_size', help='size of batch', dest='batch_size', type=int, default=32)
    parser.add_argument('-n', '--num_epochs', help='number of epochs to run', dest='num_epochs', type=int, default=20)
    parser.add_argument('-m', '--model_name', help='output filename for saved model', dest='model_name', type=str, default='model.h5')
    args = parser.parse_args()

    data_samples = load_data(args.csv_path)

    train_samples, validation_samples = train_test_split(data_samples, test_size=0.2)

#    batch_size = 32
#    num_epochs = 20
    # num_steps_per_epoch = 500
    # num_validation_steps = num_steps_per_epoch * 0.2
    num_steps_per_epoch = len(train_samples)/args.batch_size
    num_validation_steps = len(validation_samples)/args.batch_size

    train_generator = generator(train_samples, batch_size=args.batch_size)
    validation_generator = generator(validation_samples)

    model = build_model()
    model.compile(loss='mse', optimizer='adam')

    # model = load_model('model.h5')

    checkpoint = ModelCheckpoint(args.model_name, monitor='val_loss', verbose=0, save_best_only=1, mode='auto')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


    model.fit_generator(train_generator, steps_per_epoch=num_steps_per_epoch, validation_data=validation_generator, validation_steps=num_validation_steps, epochs=args.num_epochs, callbacks=[checkpoint, early_stop], verbose=1)
