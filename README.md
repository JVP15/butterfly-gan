# Butterfly Generator

This project uses GANs to generate images of butterflies.

## Dataset

We will collect our butterfly dataset using the code and techniques presented in this article: [https://marian42.de/article/butterflies/](https://marian42.de/article/butterflies/)

[This subset of the dataset](https://data.nhm.ac.uk/dataset/56e711e6-c847-4f99-915a-6894bb5c5dea/resource/05ff2255-c38a-40c9-b657-4ccb55ab2feb?view_id=6ba121d1-da26-4ee1-81fa-7da11e68f68e&filters=project%3Apapilionoidea+new+types+digitisation+project) 
contains ~1000 images of butterflies and will serve as a good start for this project.

## Model

This project uses a Generative Adversarial Network (GAN) to generate images of butterflies. Here is a [tutorial on how to create and train GANs using TensorFlow](https://www.tensorflow.org/tutorials/generative/dcgan).

After we have a working prototype using the initial butterfly dataset and a simple GAN, we can investigate using a larger dataset and [StyleGans](https://github.com/NVlabs/stylegan).