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

## Showcase
### Autoencoding (encoding -> latent space -> decoding)
![output1](https://github.com/user-attachments/assets/d82c927f-baa8-4e40-8d0c-4b7883214ea1)
![autoencoding2](https://github.com/user-attachments/assets/822ab18f-a2c5-4f6d-9993-e2d27ea6d00f)
![autoencoding3](https://github.com/user-attachments/assets/6100418c-24bf-4b1e-8f90-11c12da69e42)

### Interpolation
![interpolation1](https://github.com/user-attachments/assets/8fbbc142-f2d1-48c4-bfa0-613bda0febcf)
![interpolation2](https://github.com/user-attachments/assets/aafeb42c-8dbd-4885-9501-fdf6e5c341bb)
![interpolation3](https://github.com/user-attachments/assets/aa714dee-e5b8-41e3-9eb5-ce991dd06aaa)
![interpolation4](https://github.com/user-attachments/assets/ce2bc2d2-adb1-4c3d-b522-195d61273398)
![interpolation5](https://github.com/user-attachments/assets/35cd6c86-714f-4e2e-845a-99a3701a69fa)
![interpolation6](https://github.com/user-attachments/assets/7d31baef-ca18-4f2a-973b-4bf11744422f)
![interpolation7](https://github.com/user-attachments/assets/07f6bfad-37e6-4e74-841b-6be988769df2)
![interpolation8](https://github.com/user-attachments/assets/724c504c-9744-43fc-990a-f48f0c60b1ee)
![interpolation9](https://github.com/user-attachments/assets/ff15828f-ed24-4d3e-a15d-a66928bb5d78)

### A chart of a projection of encoded sampled instances from the learning dataset in the latent space.
![chart1](https://github.com/user-attachments/assets/8b584c67-a2e1-4944-b83b-2deb8fcb0cfe)
