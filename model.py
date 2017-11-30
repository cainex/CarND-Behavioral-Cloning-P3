import csv
import cv2
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.image as mpimg

def load_data(path):
    lines = []
    with open('{}/driving_log.csv'.format(path)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    lines.pop(0)

    return lines

def generator(path, samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '{}/IMG/'.format(path)+batch_sample[0].split('/')[-1]
                left_name = '{}/IMG/'.format(path)+batch_sample[1].split('/')[-1]
                right_name = '{}/IMG/'.format(path)+batch_sample[2].split('/')[-1]

                # center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_RGB2YUV)
#                 choice = np.random.choice(3)

#                 if choice == 0:
#                     center_image = mpimg.imread(name)
#                     # center_image = cv2.imread(name)
#                     center_angle = float(batch_sample[3])
#                     images.append(center_image)
#                     angles.append(center_angle)
#                 elif choice == 1:
# #                    left_image = cv2.cvtColor(cv2.imread(left_name), cv2.COLOR_RGB2YUV)
# #                    left_image = cv2.imread(left_name)
#                     left_image = mpimg.imread(left_name)
#                     left_angle = float(batch_sample[3]) + correction
#                     images.append(left_image)
#                     angles.append(left_angle)
#                 else:
#  #                   right_image = cv2.cvtColor(cv2.imread(left_name), cv2.COLOR_RGB2YUV)
# #                    right_image = cv2.imread(right_name)
#                     right_image = mpimg.imread(right_name)
#                     right_angle = float(batch_sample[3]) - correction
#                     images.append(right_image)
#                     angles.append(right_angle)

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
#    model.add(Lambda(lambda x: cv2.cvtColor(np.array(x), cv2.COLOR_RGB2YUV), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(24, (5, 5), activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', subsample=(2, 2)))
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

path = '/mnt/raid/projects/udacity/sdc_nd/datasets/behavioral/set_1'
samples = load_data(path)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

batch_size = 32
num_epochs = 20
# num_steps_per_epoch = 500
# num_validation_steps = num_steps_per_epoch * 0.2
num_steps_per_epoch = len(train_samples)/batch_size
num_validation_steps = len(validation_samples)/batch_size

train_generator = generator(path, train_samples, batch_size=batch_size)
validation_generator = generator(path, validation_samples)

model = build_model()

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=0, save_best_only=1, mode='auto')

model.compile(loss='mse', optimizer='Nadam')

#model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size, epochs=num_epochs, callbacks=[checkpoint], verbose=1)
model.fit_generator(train_generator, steps_per_epoch=num_steps_per_epoch, validation_data=validation_generator, validation_steps=num_validation_steps, epochs=num_epochs, callbacks=[checkpoint], verbose=1)
#model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size, epochs=num_epochs, verbose=1)
#model.fit_generator(train_generator, steps_per_epoch=num_steps_per_epoch, validation_data=validation_generator, validation_steps=num_validation_steps, epochs=num_epochs, verbose=1)

#model.save('model.h5')
