import os
import math
import numpy as np
from typing import Generator, List, Tuple, Union

import cv2

import config
# Failed to work correctly
from monochomize import monochomize_function_calls, monochomize_methods


def read_img(filename) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compatibility with DATA. Reads image and converts to Lab format.
    :param filename: File to read
    :return: Luminance, (a,b) channels of image
    """
    img = cv2.imread(filename, 3)
    labimg = cv2.cvtColor(cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
    return np.reshape(labimg[:, :, 0], (config.IMAGE_SIZE, config.IMAGE_SIZE, 1)), labimg[:, :, 1:]


class ImageDataset:
    _fileRoot: str
    _loadPercentage: float
    lastFileList: List[str] = []

    # compatibility with DATA of the old model
    filelist: List[str] = []
    batch_size: int = 0
    size: int = 0
    data_index: int = 0

    def __init__(self, folder: str = os.getcwd() + "\\Dataset", p: float = 0.1, compat_mode: bool = False):
        self._fileRoot = folder
        self._loadPercentage = p

        if compat_mode:
            self.filelist = self._get_iter_list(os.path.join(self._fileRoot, config.TRAIN_DIR))
            self.batch_size = config.BATCH_SIZE
            self.size = len(self.filelist)

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Compatibility with DATA of the old model.
        :return: grays, colors, filenames
        """
        batch = []
        labels = []
        filelist = []
        for i in range(self.batch_size):
            filename = self.filelist[self.data_index]
            filelist.append(self.filelist[self.data_index].split("\\")[-1].split("/")[-1])
            greyimg, colorimg = read_img(filename)
            batch.append(greyimg)
            labels.append(colorimg)
            self.data_index = (self.data_index + 1) % self.size
        batch = np.asarray(batch) / 255
        labels = np.asarray(labels) / 255
        return batch, labels, filelist

    def get_train_batches(self) -> int:
        folder_size = len(os.listdir(os.path.join(self._fileRoot, "train")))
        target_size = int(self._loadPercentage * folder_size)
        return int(math.ceil(target_size / float(config.BATCH_SIZE)))

    def _get_iter_list(self, folder: str) -> List[str]:
        """
        Generate list of files to load
        :param folder: Path to set
        :return:
        """
        base_set = set(os.listdir(folder))
        if self._loadPercentage >= 1.0:
            return [os.path.join(folder, f) for f in base_set]

        folder_size = len(base_set)
        target_size = int(self._loadPercentage * folder_size)
        print("selecting " + str(target_size) + "/" + str(folder_size) + " files")

        result_list: List[str] = []
        for _ in range(target_size):
            result_list.append(os.path.join(folder, base_set.pop()))

        self.lastFileList = [f.split("\\")[-1].split("/")[-1] for f in result_list]
        return result_list

    def _load_set(self, sub_folder: str, include_labels: bool)\
            -> Union[Tuple[List[np.ndarray], np.ndarray], List[np.ndarray]]:
        """
        Loads a set of images as a single set
        :param sub_folder: The subset to load
        :return: grays, downscaled grays, grays, originals
        """
        paths: List[str] = self._get_iter_list(os.path.join(self._fileRoot, sub_folder))

        print("loading " + str(len(paths)) + " images")
        img_buffer: List[np.ndarray] = [cv2.resize(cv2.imread(f), (config.IMAGE_SIZE, config.IMAGE_SIZE)) for f in paths]

        print("converting to grays")
        img_buffer = [cv2.cvtColor(i, cv2.COLOR_RGB2Lab) for i in img_buffer]
        grays: List[np.ndarray] = [np.reshape(i[:, :, 0], (config.IMAGE_SIZE, config.IMAGE_SIZE, 1)) for i in img_buffer]
        labels: List[np.ndarray] = [i[:, :, 1:] for i in img_buffer]
        img_buffer.clear()

        print("prepping secondary input")
        classifier: List[np.ndarray] = [np.reshape(cv2.resize(i, (112, 112)), (112, 112, 1)) for i in grays]

        print("converting to nparrays\n")
        if include_labels:
            return [np.asarray(grays) / 255.0, np.asarray(classifier) / 255.0], np.asarray(labels) / 255.0
        else:
            return [np.asarray(grays) / 255.0, np.asarray(classifier) / 255.0]

    def load_training_data(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load training data as a single set
        :return: grays, downscaled grays, grays, originals
        """
        return self._load_set(config.TRAIN_DIR, True)

    def load_validation_data(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load validation data as a single set
        :return: grays, downscaled grays, grays, originals
        """
        return self._load_set(config.VAL_DIR, True)

    def load_testing_data(self) -> List[np.ndarray]:
        """
        Load testing data as a single set
        :return: grays, downscaled grays, grays, originals
        """
        input_data = self._load_set(config.TEST_DIR, False)
        return input_data

    def _load_generator(self, sub_folder: str, include_labels: bool)\
            -> Generator[Union[Tuple[List[np.ndarray], np.ndarray], List[np.ndarray]], None, None]:
        """
        Loads a set of images and provides a generator for iterating them
        :param sub_folder: The subset to load
        :return: grays, downscaled grays, originals
        """
        paths: List[str] = self._get_iter_list(os.path.join(self._fileRoot, sub_folder))
        print("loading generator for " + str(len(paths)) + " images")

        def iter_paths():
            grays: List[np.ndarray] = []
            labels: List[np.ndarray] = []
            classifiers: List[np.ndarray] = []
            current_batch: int = 0

            for f in paths:
                image: np.ndarray = cv2.cvtColor(
                    cv2.resize(cv2.imread(f), (config.IMAGE_SIZE, config.IMAGE_SIZE)),
                    cv2.COLOR_RGB2Lab
                )

                gray: np.ndarray = np.reshape(image[:, :, 0], (config.IMAGE_SIZE, config.IMAGE_SIZE, 1))
                classifier: np.ndarray = np.reshape(cv2.resize(gray, (112, 112)), (112, 112, 1))

                current_batch += 1
                if include_labels:
                    labels.append(image[:, :, 1:])
                grays.append(gray)
                classifiers.append(classifier)

                # batch complete, yield
                if current_batch == config.BATCH_SIZE:
                    if include_labels:
                        yield [np.asarray(grays) / 255.0, np.asarray(classifiers) / 255.0], np.asarray(labels) / 255
                    else:
                        yield [np.asarray(grays) / 255.0, np.asarray(classifiers) / 255.0]

                    current_batch = 0
                    if include_labels:
                        labels.clear()
                    grays.clear()
                    classifiers.clear()

            if len(labels) > 0:
                # unfinished batch remaining
                if include_labels:
                    yield [np.asarray(grays) / 255.0, np.asarray(classifiers) / 255.0], np.asarray(labels) / 255
                else:
                    yield [np.asarray(grays) / 255.0, np.asarray(classifiers) / 255.0]

        if include_labels:
            for _ in range(config.NUM_EPOCHS):
                for i in iter_paths():
                    yield i
        else:
            for i in iter_paths():
                yield i

    def load_training_data_generator(self) -> Generator[Tuple[List[np.ndarray], np.ndarray], None, None]:
        """
        Load training data as a generator
        :return: grays, downscaled grays, originals
        """
        return self._load_generator(config.TRAIN_DIR, True)

    def load_testing_data_generator(self) -> Generator[List[np.ndarray], None, None]:
        """
        Load testing data as a generator
        :return: grays, downscaled grays
        """
        return self._load_generator(config.TEST_DIR, False)


def load_all_data():
    # testing purposes

    ds = ImageDataset(p=0.01)
    print("loading training data")
    ds.load_training_data()
    print("done loading")


if __name__ == "__main__":
    load_all_data()
