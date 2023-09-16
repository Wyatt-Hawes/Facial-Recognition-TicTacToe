from keras import Sequential
from keras.src.layers import RandomFlip, RandomRotation
from keras.utils import image_dataset_from_directory
from config import train_directory, test_directory, image_size, batch_size, validation_split

from matplotlib.image import imread
import tensorflow
import numpy


def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset


def _augment_dataset(dataset):
    # # YOUR CODE
    # #
    # #

    flip_and_rotate = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1)
    ])

    new_dataset = dataset.map(lambda x, y: (flip_and_rotate(x, training=True), y))
    return new_dataset


def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    # # YOUR CODE
    # # call augment_dataset
    #
    train_dataset = _augment_dataset(train_dataset)
    # validation_dataset  = _augment_dataset(validation_dataset   )
    # test_dataset        = _augment_dataset(test_dataset         )
    #
    # print("*")
    return train_dataset, validation_dataset, test_dataset


if __name__ == '__main__':
    get_datasets()
