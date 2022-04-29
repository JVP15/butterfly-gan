import os
import random
import time

import matplotlib.pyplot as plt
from pixellib.instance import custom_segmentation

import cv2
import numpy as np


class ButterflyDataset(object):
    def __init__(self, processed_image_path = 'butterflies', batch_size=32, image_size = 224, image_padding = .1):
        self.dataset_path = processed_image_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.border = image_padding # this is the % of the image that we want to pad on each side

        self.it_index = 0 # this index is used when iterating over the dataset using next()

        # try to build the dataset using existing files in the dataset folder
        if os.path.exists(self.dataset_path):
            self.filenames = self._get_img_names_in_dir(self.dataset_path)
        else:
            self.filenames = []

    def create_dataset(self, dataset_folder, num_images = None, seed=None):
        # get all the filenames in the dataset
        filenames = self._get_img_names_in_dir(dataset_folder)

        # shuffle the dataset so that the batches are different with each run
        if seed is not None:
            random.seed(seed)
        random.shuffle(filenames)
        if num_images is not None:
            filenames = filenames[:num_images]

        # create a temporary directory to store the resized images
        tempdir = os.path.join(dataset_folder, 'temp')
        os.makedirs(tempdir)

        # resize the images and save them to the temporary directory
        for i, filename in enumerate(filenames):
            img = cv2.imread(filename)
            img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
            # some of the butterfly images are flipped 90 degrees, so we have to rotate it
            img = self._correct_rotation(img)
            cv2.imwrite(os.path.join(tempdir, f'{i}.jpg'), img)

        # use pixellib to segment the images
        seg = custom_segmentation()
        seg.inferConfig(num_classes=2, class_names=["BG", "butterfly", "squirrel"]) # not sure why we need squirrel, but I think we do
        # you can find the model here: https://github.com/ayoolaolafenwa/PixelLib/releases
        seg.load_model("Nature_model_resnet101.h5")

        segmasks, outputs = seg.segmentBatch(tempdir, extract_segmented_objects=True)
        butterflies = [np.array(segmask['extracted_objects'][0], dtype=np.uint8) for segmask in segmasks]

        # create the dataset folder if it isn't already there
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        # convert the black background to a white background and save the images to the dataset folder
        for i, butterfly in enumerate(butterflies):
            butterfly[butterfly == 0] = 255
            cv2.imwrite(os.path.join(self.dataset_path, f'butterfly{i}.jpg'), butterfly)

        # create the list of filenames that we can draw on for the dataset
        self.filenames = self._get_img_names_in_dir(self.dataset_path)

        # clean up the temp directory
        for f in os.listdir(tempdir):
            os.remove(os.path.join(tempdir, f))
        os.rmdir(tempdir)

    def _get_img_names_in_dir(self, dir):
        return [os.path.join(dir, f) for f in os.listdir(dir) if f.lower().endswith('.jpg')]

    def _correct_rotation(self, img):
        # TODO: make it work for images that are flipped 90 degrees in the opposite direction (which aren't many, so it should be okay for now)
        h, w, _ = img.shape
        if h > w:
            img = cv2.transpose(img)
            img = cv2.flip(img, 0)
            h, w, _ = img.shape

        return img

    def preprocess_image(self, img):
        # start by padding the image using opencv to make it square
        h, w, _ = img.shape
        if h > w:
            border = (h - w) // 2
            img = cv2.copyMakeBorder(img, 0, 0, border, border, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        elif w > h:
            border = (w - h) // 2
            img = cv2.copyMakeBorder(img, border, border, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])

        # add another border around the image so that the butterfly isn't quite so close to the edge
        border = int(self.border * img.shape[0])
        img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # resize the image
        img = cv2.resize(img, (self.image_size, self.image_size))

        # apply canny edge detection to create the lineart
        lineart = cv2.Canny(img, 100, 200)

        # the lineart is white on black bg, but we want it to be black on white bg, so invert it
        lineart = cv2.bitwise_not(lineart)

        # lastly convert the image to rgb, since we load it with OpenCV in BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img, lineart

    def _get_image(self, path):
        # read the image
        img = cv2.imread(path)

        # preprocess the image
        img, lineart = self.preprocess_image(img)

        return img, lineart

    def _get_image_batch(self, filenames):
        # read the image and convert to RGB
        imgs = []
        linearts = []
        for f in filenames:
            img, lineart = self._get_image(f)
            imgs.append(img)
            linearts.append(lineart)
        return imgs, linearts

    def __iter__(self):
        self.it_index = 0
        return self

    def __next__(self):
        # get a batch of filenames
        filename_batch = self.filenames[self.it_index:self.it_index + self.batch_size]

        # if there are no more files to iterate over, stop iterating
        if self.it_index >= len(self.filenames):
            raise StopIteration
        else:
            self.it_index += self.batch_size

            # get the images and linearts for the images
            imgs, linearts = self._get_image_batch(filename_batch)

            return imgs, linearts

if __name__ == '__main__':
    # create the dataset
    dataset = ButterflyDataset(batch_size=4)
    # uncomment this if you want to create the dataset yourself, otherwise just use 'butterflies' from the google drive
    # dataset.create_dataset('10 reps', num_images=128)

    # display the images using a 2x2 grid
    # for _ in range(2):
    #     imgs, linearts = next(dataset)
    #     for j, img in enumerate(imgs):
    #         plt.subplot(2, 2, j + 1)
    #         plt.imshow(img)
    #         plt.axis('off')
    #     plt.show()
    #     # Save the linearts





        # for j, linearts in enumerate(linearts):
        #     plt.subplot(2, 2, j + 1)
        #     plt.imshow(linearts, cmap='gray')
        #     plt.axis('off')

        # plt.show()

    # Save each lineart image

    # Check if lineart direcotry exists
    path = 'lineart'
    if not os.path.exists(path):
        os.makedirs(path)

    # Iterate through the dataset
    imgs, linearts = next(dataset)

    for i, lineart in enumerate(linearts):
        cv2.imwrite(os.path.join(path, f'lineart{i}.jpg'), lineart)

    # Save each butterfly image

    # path = 'butterflies'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    
    # for i, butterfly in enumerate(dataset):
    #     cv2.imwrite(os.path.join(path, f'butterfly{i}.jpg'), butterfly)
    
