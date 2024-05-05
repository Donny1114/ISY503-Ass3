import sys
import csv
import os
import pathlib
import argparse
from datetime import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from pandas import Series
from keras.layers import Conv2D, Dense, Dropout, Flatten, Rescaling
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential
from keras.losses import MSE
from keras.optimizers import Adam
from PIL import Image, ImageFilter, ImageOps

# Print the TensorFlow version installed
print(tf.__version__)

# Specify dimensions and cropping parameters for images
origin_colours: int = 3
origin_image_height: int = 160
origin_image_width: int = 320
crop_right: int = 0
crop_left: int = 0
crop_top: int = 55
crop_bottom: int = 25

# Functions for image processing and augmentation
def crop(img: Image) -> Image:
    """Crops the image to remove irrelevant information for steering angle predictions """

    return img.crop((
        crop_left,
        crop_top,
        origin_image_width - crop_right,
        origin_image_height - crop_bottom,
    ))


def cropped_height() -> int:
    """This function returns the height of the cropped image."""
    return origin_image_height - crop_top - crop_bottom


def cropped_width() -> int:
    """Crops the image to remove irrelevant information for steering angle predictions."""
    return origin_image_width - crop_left - crop_right


def flip_horizontally(img: Image) -> Image:
    """ This function is used for dataset augmentation. It flips input image horizontally. """

    flipped = ImageOps.mirror(img)

    if options.debug and np.random.rand() < 0.001:
        img.save("debug_augmentation_flip_origin.jpg")
        flipped.save("debug_augmentation_flip_processed.jpg")

    return flipped


def blur(img: Image) -> Image:
    """ The purpose of this function is to enrich datasets. The Gaussian Blur algorithm is used to blur the input image,
     is not enabled by default in the present implementation, but it is simple to enable in `model.get_datasets_from_logs}
      if necessary.
     """
    blurred = img.filter(ImageFilter.GaussianBlur(3))

    if options.debug and np.random.rand() < 0.001:
        img.save("debug_augmentation_blur_origin.jpg")
        blurred.save("debug_augmentation_blur_processed.jpg")

    return blurred


def grayscale(img: Image) -> Image:
    """ The purpose of this function is to enrich datasets. Images submitted are converted to grayscale. is not enabled
     by default in the present implementation, but it is simple to enable in `model.get_datasets_from_logs} if necessary.
    """

    grayed = ImageOps.grayscale(img)

    if options.debug and np.random.rand() < 0.001:
        img.save("debug_augmentation_grayscale_origin.jpg")
        grayed.save("debug_augmentation_grayscale_processed.jpg")

    return grayed


def three_dimensional_grayscale(img: Image) -> Image:
    """ To use with actual RGB images, this method turns a 1-channel grayscale image into a 3-channel grayscale image.
    """
    return Image.merge('RGB', (img, img, img))


def equalize(img: Image) -> Image:
    """ The image's histogram is equalized, or normalized, using this function.
    """
    equalized = ImageOps.equalize(img)

    if np.random.rand() < 0.001:
        img.save("debug_augmentation_equalize_origin.jpg")
        equalized.save("debug_augmentation_equalize_processed.jpg")

    return equalized

# Function for saving autonomous images for further training
def save_autonomous_image(path: str, img: Image, steering: float) -> None:
    """Saves an image received in autonomous mode to the specified directory for further training."""
    img_subdir: str = "IMG"
    write_mode = "a"

    if not os.path.exists(os.path.join(path, img_subdir)):
        pathlib.Path(os.path.join(path, img_subdir)).mkdir(parents=True, exist_ok=True)
        write_mode = "w"

    basename = dt.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + ".jpg"
    img.save(os.path.join(path, img_subdir, basename))

    with open(os.path.join(path, "driving_log.csv"), write_mode) as fd:
        writer = csv.writer(fd)
        writer.writerow([img_subdir + "/" + basename, "", "", str(steering)])
        fd.close()

# Function for checking if a row belongs to autonomous or human-gathered data
def is_autonomous_row(row: Series) -> bool:
    """Preprocesses, augments, and returns training and validation datasets based on driving logs."""
    return pd.isna(row['right']) and pd.isna(row['left'])


def get_unit_of_data_from_autonomous_data(row: Series, steering: float, extra_angle: float) -> (Image, float):
    """Preprocesses, augments, and returns training and validation datasets based on driving logs."""
    if is_autonomous_row(row):
        image = Image.open(row['center'])
    else:
        match np.random.choice(2):
            case 0:
                image = Image.open(row['left'])
                steering += extra_angle
            case 1:
                image = Image.open(row['right'])
                steering -= extra_angle
            case _:
                raise Exception("unexpected choice")

    return image, steering


def get_unit_of_data_from_human_gathered_data(row: Series, steering: float, extra_angle: float) -> (Image, float):
    """ One image is returned by this function for each record in the file {{driving_log.csv}}. It should only be used to datasets that were collected by humans and contain
    all three photos (center, left, and right).
    """
    match np.random.choice(3):
        case 0:
            image = Image.open(row['center'])
        case 1:
            image = Image.open(row['left'])
            steering += extra_angle
        case 2:
            image = Image.open(row['right'])
            steering -= extra_angle
        case _:
            raise Exception("unexpected choice")

    return image, steering


def get_driving_logs(dirs: list[str]) -> pd.DataFrame:
    """ This function creates a single virtual dataset by combining a list of the relative paths to read their contents ({{driving_log.csv}{). It assists in compiling several independent datasets, combining them during training, determining
    the optimal combination of them, and eliminating sub-datasets as they become unnecessary.

    """
    clear_data_list: list[pd.DataFrame] = []

    for dir in dirs:
        print("Reading " + dir, file=sys.stderr)

        csv = pd.read_csv(
            os.path.join(dir, 'driving_log.csv'),
            delimiter=',',
            names=['center', 'left', 'right', 'steering'],
            usecols=[0, 1, 2, 3],
        )

        for column in ['center', 'left', 'right']:
            if csv[column].count() > 0:
                separator = "/" if "/" in csv[column][0] else "\\"
                csv[column] = csv[column].map(lambda path: os.path.join(dir, *path.split(separator)[-2:]))

        clear_data_list.append(csv)

    return pd.concat(clear_data_list)


def get_datasets_from_logs(logs: pd.DataFrame, autonomous: bool, validation_data_percent: float, extra_angle: float) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """ This is among the primary functions. A panda is required.Get a DataFrame object from {{get_driving_logs}}, then return a validation dataset and a preprocessed and augmented t
    raining data set. The `{validation_data_percent}} parameter controls the ratio between them.
    """
    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []

    val_x: list[np.ndarray] = []
    val_y: list[np.ndarray] = []

    for index, row in logs.iterrows():
        steering = row['steering']

        if autonomous:
            image, steering = get_unit_of_data_from_autonomous_data(row, steering, extra_angle)
        else:
            image, steering = get_unit_of_data_from_human_gathered_data(row, steering, extra_angle)

        training_image = np.random.rand() > validation_data_percent

        if training_image:
            if np.random.rand() < 0.5:
                image = flip_horizontally(image)
                steering *= -1



        image = image.crop((
            crop_left,
            crop_top,
            origin_image_width - crop_right,
            origin_image_height - crop_bottom,
        ))

        image = equalize(image)

        # image = add_gray_layer_to_rgb_image(image)

        image = np.asarray(image)

        if training_image:
            train_x.append(image)
            train_y.append(steering)
        else:
            val_x.append(image)
            val_y.append(steering)

    return np.asarray(train_x), np.asarray(train_y), np.asarray(val_x), np.asarray(val_y)


def build_model() -> Sequential:
    """ This function creates a CNN model that is utilized to forecast steering angles. It has an input layer,
        many hidden layers, an output layer, dropouts to prevent overfitting, and multiple convolution layers.

    :return:
    """
    model = Sequential()

    # The first layer rescales input values from [0, 255] format to [-1, 1]
    model.add(Rescaling(1.0/127.5, offset=-1, input_shape=(cropped_height(), cropped_width(), origin_colours)))
    # Several layers to convolve input data from (320, 80, 3) shape to (1, 15, 96) feature maps along with rectified
    # linear unit (ReLU) activation functions
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(3, 3), activation="relu"))
    model.add(Conv2D(filters=48, kernel_size=(3, 3), strides=(3, 3), activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(filters=96, kernel_size=(3, 3), activation="relu"))
    model.add(Dropout(0.25))
    # Input layer
    model.add(Flatten())  # 1440 pixels
    model.add(Dropout(0.25))
    # Hidden layers with ReLU activation functions
    model.add(Dense(units=500, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=100, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=50, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=10, activation="relu"))
    # Output layer
    model.add(Dense(units=1))

    # To compute errors between labels and predictions, use the mean squared error (MSE) loss function.
    # Based on adaptive estimate of first-order and #second-order moments,
    # Adam optimization is a randomized gradient descent technique.
    model.compile(loss=MSE, optimizer=Adam(learning_rate=0.001))

    return model


def draw_plot(iterations, *args):
    for i in range(0, len(args)-1, 2):
        plt.plot(range(1, iterations + 1), args[i], label=args[i+1])

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='best', fontsize='small')

    plt.savefig("Loss history.jpg")


def model_callback_list() -> list[Callback]:
    list: list[Callback] = []

    list.append(
        # Saves model to disk once validation loss after this epoch is lower than after previous one
        ModelCheckpoint(
            "model-{}.h5".format(dt.now().strftime("%Y-%m-%d-%H-%M-%S")),
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True,
        ),
    )

    return list


if __name__ == '__main__':
    cli_opts = argparse.ArgumentParser()
    cli_opts.add_argument('--debug', default=True, action='store_true', help='Debug mode')
    cli_opts.add_argument('--sources', nargs='+', help='Path to datasets: --sources Track-1/f1 Track-1/b1', required=True)
    cli_opts.add_argument('--train-on-autonomous-center', default=False, action='store_true', help='Whether to use only autonomous center images or not')
    cli_opts.add_argument('--print-only', default=False, action='store_true', help='Print information on layers end exit')
    cli_opts.add_argument('--epochs', type=int, default=10, help='Number of epochs of training')
    cli_opts.add_argument('--validation-data-percent', type=float, default=0.3, help='The size of validation dataset [0, 1]')
    cli_opts.add_argument('--extra-angle', type=float, default=0.2, help='This extra value will be added when the car diverges from the center')
    options = cli_opts.parse_args()

    # Builds the CNN model for training
    model = build_model()

    if options.print_only:
        print(model.summary())
        exit(0)

    # Gathers data from driving_log.csv files into a single pandas file.DataFrame element.
    logs = get_driving_logs(options.sources)

    # Combines one or more sub-datasets to perform preprocessing and augmentation,
    # and then outputs a list of training and validation datasets.
    train_X, train_Y, val_X, val_Y = get_datasets_from_logs(
        logs,
        options.train_on_autonomous_center,
        options.validation_data_percent,
        options.extra_angle,
    )

    # Uses datasets to train the build model.
    history = model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=options.epochs, callbacks=model_callback_list())

    # Retains a historical graph of the validation and training losses
    draw_plot(
        history.params['epochs'],
        history.history['val_loss'], 'Validation Loss',
        history.history['loss'], 'Training loss',
    )
