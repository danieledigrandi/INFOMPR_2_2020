import os
import numpy as np
from typing import List, Optional

import cv2
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

from tensorflow.keras.layers import Conv2D, Dense, Input, Layer, UpSampling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError


import config
from dataset import ImageDataset
# Failed to work correctly
from monochomize import monochomize_function_names, monochomize_methods


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY, filelist):
    for i in range(config.BATCH_SIZE):
        result = np.concatenate((batchX[i], predictedY[i]), axis=2)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        save_path = os.path.join(config.OUT_DIR, filelist[i][:-4] + "reconstructed.jpg")
        cv2.imwrite(save_path, result)


class FusionLayer(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):
        mid_features = inputs[0]
        global_features = inputs[1]

        mid_features_shape = mid_features.get_shape().as_list()
        mid_features_reshaped = tf.reshape(
            mid_features,
            [
                -1,
                mid_features_shape[1] * mid_features_shape[2],
                mid_features_shape[3]
            ]
        )

        # repeat global output layer for each "pixel"
        repeat = tf.constant([1, mid_features_shape[1] * mid_features_shape[2]], tf.int32)
        global_rep = tf.tile(global_features, repeat)
        global_rep = tf.reshape(global_rep, (-1, mid_features_shape[1] * mid_features_shape[2], 256))

        # interleave networks
        stack = tf.stack([mid_features_reshaped, global_rep], axis=2)
        return tf.reshape(
            stack,
            [
                -1,
                mid_features_shape[1],
                mid_features_shape[2],
                mid_features_shape[3] + global_features.get_shape().as_list()[1]
            ]
        )


class MODEL:
    input_img: Input
    input_net: Input
    model: Optional[Model]

    def __init__(self):
        self.input_img = Input((config.IMAGE_SIZE, config.IMAGE_SIZE, 1))
        self.input_net = Input((112, 112, 1))
        self.model = None

    def build(self):
        low_level: List[Layer] = [
            Conv2D(64, (3, 3), (2, 2), padding="same", activation="relu"),
            Conv2D(128, (3, 3), (1, 1), padding="same", activation="relu"),
            Conv2D(128, (3, 3), (2, 2), padding="same", activation="relu"),
            Conv2D(256, (3, 3), (1, 1), padding="same", activation="relu"),
            Conv2D(256, (3, 3), (2, 2), padding="same", activation="relu"),
            Conv2D(512, (3, 3), (1, 1), padding="same", activation="relu")
        ]

        mid_level: List[Layer] = [
            Conv2D(512, (3, 3), (1, 1), padding="same", activation="relu"),
            Conv2D(256, (3, 3), (1, 1), padding="same", activation="relu")
        ]

        global_level: List[Layer] = [
            Conv2D(512, (3, 3), (2, 2), padding="same", activation="relu"),
            Conv2D(512, (3, 3), (1, 1), padding="same", activation="relu"),
            Conv2D(512, (3, 3), (2, 2), padding="same", activation="relu"),
            Conv2D(512, (3, 3), (1, 1), padding="same", activation="relu"),
            Flatten(),
            Dense(1024, activation="relu"),
            Dense(512, activation="relu"),
            Dense(256, activation="relu")
        ]

        fusion_level: List[Layer] = [
            Conv2D(256, (3, 3), (1, 1), padding="same", activation="relu"),
            Conv2D(128, (3, 3), (1, 1), padding="same", activation="relu")
        ]

        colourise: List[Layer] = [
            UpSampling2D(),
            Conv2D(64, (3, 3), (1, 1), padding="same", activation="relu"),
            Conv2D(64, (3, 3), (1, 1), padding="same", activation="relu"),
            UpSampling2D(),
            Conv2D(32, (3, 3), (1, 1), padding="same", activation="relu"),
            Conv2D(2, (3, 3), (1, 1), padding="same", activation="relu"),
            UpSampling2D()
        ]

        colour_input = self.input_img
        for layer in low_level:
            colour_input = layer(colour_input)
        for layer in mid_level:
            colour_input = layer(colour_input)

        classification = self.input_net
        for layer in low_level:
            classification = layer(classification)
        for layer in global_level:
            classification = layer(classification)

        colour_net = FusionLayer()([colour_input, classification])
        for layer in fusion_level:
            colour_net = layer(colour_net)
        for layer in colourise:
            colour_net = layer(colour_net)

        colour_model = Model([self.input_img, self.input_net], [colour_net])
        colour_model.compile(optimizer="adam", loss=MeanSquaredError())

        colour_model.summary()
        self.model = colour_model

    def train(self, data: ImageDataset):
        # set based training
        # train_data, train_label = data.load_training_data()
        # val_data = data.load_validation_data()
        # self.model.fit(
        #     train_data,
        #     train_label,
        #     validation_data=val_data,
        #     epochs=config.NUM_EPOCHS,
        #     batch_size=config.BATCH_SIZE
        # )

        # generator based training
        self.model.fit(
            data.load_training_data_generator(),
            epochs=config.NUM_EPOCHS,
            steps_per_epoch=data.get_train_batches()
        )

    def test(self, data: ImageDataset):
        test_data = data.load_testing_data()
        predictions = self.model.predict(test_data)
        grays, _ = test_data
        for i in range(len(data.lastFileList)):
            result = np.concatenate((grays[i], predictions[i]), axis=-1)
            result = deprocess(result)
            result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
            save_path = os.path.join(
                config.OUT_DIR,
                data.lastFileList[i][:-4] + "_reconstructed.jpg"
            )
            cv2.imwrite(save_path, result)
