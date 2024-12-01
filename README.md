# VAE GAN

## About
This repository contains an implementation of a Variational Autoencoder (VAE) combined with a Generative Adversarial Network (GAN) in PyTorch. The model is trained on the Flowers dataset to generate realistic floral images.

Architecture:
- `VAE Encoder`: Maps input images to a latent space.
- `VAE Decoder`: Reconstructs images from latent representations.
- `GAN Discriminator`: Ensures generated images are indistinguishable from real ones.

## Installation
In order to run the program, you will need to install Python3.13 and pip. Then install the required dependencies by typing:
```bash
pip install -r requirements.txt
```
