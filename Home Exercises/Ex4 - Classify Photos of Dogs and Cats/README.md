# Deep Neural Network Image Classifier 🌌

Hello there, dear reader! 👋 This repository is more than just a collection of Python scripts. It's a testament to the power of deep learning, PyTorch, and of course, my passion for Machine Learning. If you've ever wondered about building a robust image classifier using PyTorch, then you're in for a treat! 🍩

## Overview 🌐

The heart of this repository is a deep neural network (DNN) that's trained to classify images. Built with several convolutional layers, this DNN is not just a toy model - it is a sophisticated and powerful beast capable of making sense out of complex visual data.

## Features 🔥

- **Image Resizer**: Built with PIL, this utility resizes images into a desired dimension ensuring the aspect ratio remains undistorted.
  
- **Dynamic Data Augmentation**: Uses random rotations, flips, and normalization, ensuring the model never sees the exact same picture twice during training, making it robust!

- **Deep Neural Network Architecture**: A seven-layered convolutional neural network with dropout and batch normalization to ensure efficient and fast learning.

- **Custom Training Loop**: With detailed logging, ensuring you know how your model is performing at every epoch.

- **Data Handling**: Automatic data loading and splitting functionalities for training and testing datasets using PyTorch's DataLoader.

## Quickstart 🚀

1. Clone this repository.
2. Ensure you have the necessary libraries installed (PyTorch, torchvision, PIL, etc.).
3. Prepare your image data set. The current code expects folders named after classes filled with relevant images.
4. Run the script! 🎉

## Results & Visualization 📊

At the end of the training, you'll be presented with a graph plotting the training and validation losses across epochs. It's not just about achieving high accuracy; it's about understanding the journey there!

## Why this Repository? 🤔

It's simple. I have a passion for deep learning, and I believe in its transformative power. This repository stands as a testament to my commitment, my attention to detail, and my eagerness to make an impact in the field of AI.

## Future Plans 🛠️

- Implementing checkpoints and saving the best model.
- Extending the model to handle more complex datasets.
- Real-time validation with test-time augmentation.

## Let's Talk! 💌

If you find this repository impressive, let's discuss it further. I'd be thrilled to delve into the intricacies of the code, the design decisions made, and the potential improvements. 
