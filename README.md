# CIFAR-10 Image Classification with CNN

This repository contains a Convolutional Neural Network (CNN) model trained to classify images from the CIFAR-10 dataset.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

The 10 classes are:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck


## Model

The CNN model used in this project is a simple architecture with two convolutional layers followed by two fully connected layers.

- **Convolutional Layers:** The convolutional layers extract features from the input images.
- **Fully Connected Layers:** The fully connected layers classify the images based on the extracted features.

## Training

The model is trained using the following hyperparameters:

- **Number of Epochs:** 10
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy Loss

The model is trained on the training set of the CIFAR-10 dataset and evaluated on the test set.

## Results

The model achieves an accuracy of approximately **71.06%** on the test set.

## Usage

To run the code:

1. Make sure you have the necessary libraries installed (PyTorch, torchvision, etc.).
2. Download the CIFAR-10 dataset.
3. Run the Python script to train and evaluate the model.

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
