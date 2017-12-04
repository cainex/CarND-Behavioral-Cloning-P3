# from keras import backend as K
# from keras.layers import Input, Cropping2D
# from keras.models import Sequential
import argparse
from utils import load_data, generator
import matplotlib.pyplot as plt 
import numpy as np
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Self-driving car simulation training script')
    parser.add_argument('-p', '--path', help='path to dataset CSV', dest='csv_path', type=str)
    args = parser.parse_args()

    print('Loading samples from CSV...')
    samples = load_data(args.csv_path)
    print('Getting next images...')
    gen = generator(samples, 1, False)
    for i in range(0,100):
        X_train, y_train = next(gen)
    print(X_train.shape)

    fig, ax = plt.subplots(ncols=2, nrows=3)

    ax[0][0].imshow(X_train[0])
    ax[0][0].set_title(y_train[0])
    ax[0][0].set_xlabel('center')
    ax[0][1].imshow(X_train[1])
    ax[0][1].set_title(y_train[1])
    ax[0][1].set_xlabel('center flipped')
    ax[1][0].imshow(X_train[2])
    ax[1][0].set_title(y_train[2])
    ax[1][0].set_xlabel('left')
    ax[1][1].imshow(X_train[3])
    ax[1][1].set_title(y_train[3])
    ax[1][1].set_xlabel('left flipped')
    ax[2][0].imshow(X_train[4])
    ax[2][0].set_title(y_train[4])
    ax[2][0].set_xlabel('right')
    ax[2][1].imshow(X_train[5])
    ax[2][1].set_title(y_train[5])
    ax[2][1].set_xlabel('right flipped')

    #plt.show()
    plt.tight_layout()
    plt.savefig('images/images.png')

    y_length = X_train.shape[1]
    x_length = X_train.shape[2]

    cropped_img = []
    for i in range(0,X_train.shape[0]):
        tmp_img = X_train[i]
        cropped_img.append(tmp_img[50:y_length-25,0:x_length,0:3])

    fig, ax = plt.subplots(ncols=2, nrows=3)

    ax[0][0].imshow(cropped_img[0])
    ax[0][0].set_title(y_train[0])
    ax[0][0].set_xlabel('center')
    ax[0][1].imshow(cropped_img[1])
    ax[0][1].set_title(y_train[1])
    ax[0][1].set_xlabel('center flipped')
    ax[1][0].imshow(cropped_img[2])
    ax[1][0].set_title(y_train[2])
    ax[1][0].set_xlabel('left')
    ax[1][1].imshow(cropped_img[3])
    ax[1][1].set_title(y_train[3])
    ax[1][1].set_xlabel('left flipped')
    ax[2][0].imshow(cropped_img[4])
    ax[2][0].set_title(y_train[4])
    ax[2][0].set_xlabel('right')
    ax[2][1].imshow(cropped_img[5])
    ax[2][1].set_title(y_train[5])
    ax[2][1].set_xlabel('right flipped')

    #plt.show()
    plt.tight_layout()
    plt.savefig('images/cropped_images.png')
