import csv
import numpy as np
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

def generator(samples, batch_size=32, do_shuffle=True):
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
            if do_shuffle:
                X_train, y_train = shuffle(X_train, y_train)
            yield X_train, y_train

