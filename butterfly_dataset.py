import os
import random
import matplotlib.pyplot as plt

import cv2


class ButterflyDataset(object):
    def __init__(self, data_path, batch_size, seed=None):
        self.data_path = data_path
        self.batch_size = batch_size

        # get all the filenames in the dataset
        self.filenames = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jpg')]

        # shuffle the dataset so that the batches are different with each run
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.filenames)

    def _get_image(self, path):
        # read the image and convert to RGB
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _get_image_batch(self, filenames):
        # read the image and convert to RGB
        imgs = []
        for f in filenames:
            img = self._get_image(f)
            imgs.append(img)
        return imgs

    def __iter__(self):
        return self

    def __next__(self):
        # get a batch of filenames
        filename_batch = self.filenames[:self.batch_size]

        # if there are no more files to iterate over, stop iterating
        if len(filename_batch) == 0:
            raise StopIteration
        else: # otherwise, get the iamges and remove the files from the list
            self.filenames = self.filenames[self.batch_size:]

            # get the images
            imgs = self._get_image_batch(filename_batch)

            return imgs

if __name__ == '__main__':
    # create the dataset
    dataset = ButterflyDataset('10 reps', batch_size=4)

    # get the next batch of images
    batch = next(dataset)
    batch2 = next(dataset)

    # display the images using a 2x2 grid
    for b in [batch, batch2]:
        for j, img in enumerate(b):
            plt.subplot(2, 2, j + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()

