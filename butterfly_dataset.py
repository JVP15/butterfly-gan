import os
import random
import shutil
import time

import matplotlib.pyplot as plt

import cv2
import numpy as np


class ButterflyDataset(object):
    def __init__(self, processed_image_path = 'butterflies', batch_size=256, image_size = 224, image_padding = .1, fill_lineart = False,
                 invert_lineart = True):
        self.dataset_path = processed_image_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.border = image_padding # this is the % of the image that we want to pad on each side
        self.fill_lineart = fill_lineart
        self.invert_lineart = invert_lineart

        self.it_index = 0 # this index is used when iterating over the dataset using next()

        # try to build the dataset using existing files in the dataset folder
        if os.path.exists(self.dataset_path):
            self.filenames = self._get_img_names_in_dir(self.dataset_path)
        else:
            self.filenames = []

    def create_dataset(self, dataset_folder = '10 reps', num_images = None, seed=None):
        """Download the dataset from here: https://zenodo.org/record/4307612#.Ym2CrtrMKUk and extract it to a
        folder named '10 reps'. This function will create a folder named 'butterflies' that contains some preprocessed
        images of butterflies on blank white backgrounds. NOTE: this function uses pixellib, which only supports
        TensorFlow 2.0.0 to 2.4.1, so you will need to use one of those versions. """
        from pixellib.instance import custom_segmentation

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
        batchdir = os.path.join(tempdir, 'batch')
        os.makedirs(tempdir, exist_ok=True)
        os.makedirs(batchdir, exist_ok=True)

        # resize the images and save them to the temporary directory
        for i, filename in enumerate(filenames):
            print(f'Resizing image {i} of {len(filenames)}\r', end='')
            img = cv2.imread(filename)
            img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
            # some of the butterfly images are flipped 90 degrees, so we have to rotate it
            img = self._correct_rotation(img)
            cv2.imwrite(os.path.join(tempdir, f'{i}.jpg'), img)

        filenames = [f for f in os.listdir(tempdir) if f.lower().endswith('.jpg')]

        print('\nSegmenting images')
        # use pixellib to segment the images
        seg = custom_segmentation()
        seg.inferConfig(num_classes=2,
                        class_names=["BG", "butterfly", "squirrel"])  # not sure why we need squirrel, but I think we do
        # you can find the model here: https://github.com/ayoolaolafenwa/PixelLib/releases
        seg.load_model("Nature_model_resnet101.h5")

        # seperate the images into batches so that pixellib only processes a small number of images at a time
        for i in range(0, len(filenames), self.batch_size):
            print(f'Segmenting batch {i} of {len(filenames) // self.batch_size}\r', end='')
            filename_batch = filenames[i:i+self.batch_size]

            for filename in filename_batch:
                src_filename = os.path.join(tempdir, filename)
                dst_filename = os.path.join(batchdir, filename)
                shutil.move(src_filename, dst_filename)

            segmasks, outputs = seg.segmentBatch(tempdir, extract_segmented_objects=True)
            butterflies = [np.array(segmask['extracted_objects'][0], dtype=np.uint8) for segmask in segmasks]

            # create the dataset folder if it isn't already there
            if not os.path.exists(self.dataset_path):
                os.makedirs(self.dataset_path)

            # convert the black background to a white background and save the images to the dataset folder
            for i, butterfly in enumerate(butterflies):
                butterfly[butterfly == 0] = 255
                cv2.imwrite(os.path.join(self.dataset_path, f'butterfly{i}.jpg'), butterfly)

            # remove all of the images in the batch folder
            for filename in filename_batch:
                os.remove(os.path.join(batchdir, filename))

        print('\nSegmented Images')
        # create the list of filenames that we can draw on for the dataset
        self.filenames = self._get_img_names_in_dir(self.dataset_path)

        # clean up the temp directory
        shutil.rmtree(tempdir)

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
        lineart = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 230, 250)

        #lineart = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY_INV)[1]
        # find the external contours of the lineart and then draw them on the on a black image
        """Idea: find all contours from canny edge detection, but only keep, like, the the largest ones"""
        contours, _ = cv2.findContours(lineart, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        num_contours = max(6, int(np.sqrt(len(sorted_contours))))

        sorted_contours = sorted_contours[:num_contours]

        lineart = np.zeros(lineart.shape, dtype=np.uint8)
        cv2.drawContours(lineart, sorted_contours, -1, (255, 255, 255), 2)

        # the lineart is white on black bg, but normally we want it to be black on white bg, so invert it
        # if self.invert_lineart:
        #     lineart = cv2.bitwise_not(lineart)

        # convert the image to rgb, since we load it with OpenCV in BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # we've found out that GANs mostly just modify non-white parts of the image, so the results could be improved
        #   by filling in the line art with gray pixels, so we still see the edge of the image, but the butterfly is not all white
        if self.fill_lineart:
            mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            mask[mask < 255] = 200
            mask[lineart == 0] = 0
            lineart = mask

        return img, lineart

    def save_to_folder(self, folder='butterflies', new_img_size = None):
        """Saves the preprocessed images to a folder named 'processed' and the lineart to a folder named 'lineart'.
        By default, these folders are created in the 'butterflies' directory, so you would have '
        butterflies/processed' and 'butterflies/lineart'. """

        lineart_folder = os.path.join(folder, 'lineart')
        processed_folder = os.path.join(folder, 'processed')

        if not os.path.exists(lineart_folder):
            os.makedirs(lineart_folder)
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)

        for i, filename in enumerate(self.filenames):
            img = cv2.imread(filename)
            img, lineart = self.preprocess_image(img)
            cv2.imwrite(os.path.join(lineart_folder, f'{i}.jpg'), lineart)

            # when we preprocess the image, we convert it to RGB, so we need to convert it back to BGR to save it with OpenCV
            cv2.imwrite(os.path.join(processed_folder, f'{i}.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

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

    def get_images(self):
        """Returns all of the preprocessed and line art images in the dataset in the form: imgs: List, lineart: List"""
        return self._get_image_batch(self.filenames)

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
    dataset = ButterflyDataset(invert_lineart=False, image_size=256)
    #dataset.create_dataset('10 reps')
    dataset.save_to_folder()

    # butterfly_imgs_and_linearts = dataset.get_images()
    # # display the images next to their line arts using a 2x2 grid in matplotlib
    # num_imgs = 3
    # imgs, linearts = butterfly_imgs_and_linearts[:num_imgs]
    # fig, axs = plt.subplots(num_imgs, 2)
    # for i, ax in enumerate(axs.flat):
    #     if i % num_imgs == 0:
    #         ax.imshow(imgs[i // num_imgs])
    #     else:
    #         ax.imshow(linearts[i // num_imgs], cmap='gray')
    #     ax.axis('off')
    # plt.tight_layout()
    # plt.show()



