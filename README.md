# Butterfly Colorization

This project uses GANs to color butterfly images. In `butterfly_color_notebook.ipynb`, we attempt to colorize grayscale butterfly images. In `butterly_pix2pix_notebook.ipynb`, we attempt to recreate the original butterfly image from the line art of that image. 

## Installation

Before running our code, you need to install the dependancies. You can do this by running:

`pip install -r requirements.txt`

If there are any errors with `pip`, you may try variants such as `pip3`, or `python3 -m pip` instead. Optionally, you could create an environment using `venv` or `conda` before running that command so that you have a fresh environment.

## Dataset

We have provided a partially-preprocessed dataset for your convenience. The file `butterflies.zip` contains 128 preprocessed butterfly images. All you have to do to load the dataset is execute the command:

`unzip butterflies.zip`

We got our original dataset from [Representative images of Lepidoptera from the Natural History Museum](https://zenodo.org/record/4307612#.Ym2CrtrMKUk). We then applied some preprocessing steps in `butterfly_dataset.py`. If you want to recreate our steps exactly, you can download the dataset yourself and execute `python butterfly_dataset.py`. However, there are many known complications (see `ButterflyDataset.create_dataset()` for more info) when creating the dataset from scratch. For the sake of simplicity, just use the dataset in `butterflies.zip`. 

## Coloring Butterfly Images

You can open the butterfly colorization notebook by running the command:

`jupyterlab butterfly_color_notebook.ipynb`

To run the program, click on the `Restart kernel and run all cells` button on the toolbar, or hit {CTRL} + {ENTER} on your keyboard to run all of the cells. Depending on your GPU, the notebook could take 1 hour or more to run. 

## Using Pix2Pix to Convert Line Art to Butterflies

You can open the line art to butterfly notebook by running the command:

`jupyterlab butterfly_pix2pix_notebook.ipynb`

To run the program, click on the `Restart kernel and run all cells` button on the toolbar, or hit {CTRL} + {ENTER} on your keyboard to run all of the cells. Depending on your GPU, the notebook could take 1 hour or more to run. 


## References

### Dataset / Preprocessing
* https://zenodo.org/record/4307612#.Ynrx3fPMI1J
* https://github.com/ayoolaolafenwa/PixelLib 

### GAN 
* https://www.tensorflow.org/tutorials/generative/dcgan
 
### Colorization
* https://www.youtube.com/watch?v=v88IUAsgfz0
* https://github.com/OvaizAli/Image-Colorization-using-GANs

### Line-art Interpolation 
* https://phillipi.github.io/pix2pix/
* https://www.tensorflow.org/tutorials/generative/pix2pix


