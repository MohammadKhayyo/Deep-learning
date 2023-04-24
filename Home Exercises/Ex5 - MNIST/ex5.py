import numpy as np
import torch
from torchvision.transforms import transforms
import torch.nn as nn
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from copy import deepcopy


class Autoencoder(nn.Module):
    def __init__(self, noise=False, level_of_noise=0.4, epochs=100, batchSize=128, learningRate=1e-3):
        """This code defines the constructor for the Autoencoder class. It initializes the encoder and decoder
        networks as a series of Conv2d and ConvTranspose2d layers with ReLU activation functions. It also sets the
        device to use for training (either GPU or CPU) and initializes the hyperparameters such as the number of
        epochs, batch size, learning rate, and noise factor. It then creates the optimizer (Adam) and loss function (
        MSE loss) and loads the MNIST training and test datasets using PyTorch's datasets module and DataLoader
        class. Finally, it initializes empty lists for storing the average training and test losses during training. """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.noise = noise
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)  # , weight_decay=1e-5
        self.train_mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(self.train_mnist_data,
                                                        batch_size=self.batchSize,
                                                        shuffle=True)
        self.test_mnist_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
        self.test_loader = torch.utils.data.DataLoader(self.test_mnist_data,
                                                       batch_size=self.batchSize,
                                                       shuffle=True)
        self.level_of_noise = level_of_noise
        self.average_train_loss, self.average_test_loss = list(), list()

    def forward(self, x):
        """This method defines the forward pass of the autoencoder. It takes in an input tensor and applies the
        encoder and decoder networks to it, returning the reconstructed output. """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train(self, **kwargs):
        """This method trains the autoencoder model on the training data for a specified number of epochs. At each
        epoch, it iterates through the training data, applies the model to the input images, computes the loss,
        and performs backpropagation and optimization. It also computes and stores the average training and test
        losses for each epoch."""
        torch.manual_seed(42)
        for epoch in range(self.epochs):
            train_loss_per_epoch = list()
            for Image, label in self.train_loader:
                imgIn2Autoencoder = torch.clone(Image)
                if self.noise:
                    imgIn2Autoencoder = imgIn2Autoencoder + self.level_of_noise * np.random.normal(
                        size=imgIn2Autoencoder.shape).astype(
                        "float32")
                Image = Image.to(self.device)
                imgIn2Autoencoder = imgIn2Autoencoder.to(self.device)
                OutPut = self(imgIn2Autoencoder)
                loss = self.criterion(OutPut, Image)
                train_loss_per_epoch.append(float(loss))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.average_train_loss.append(np.array(train_loss_per_epoch).mean())
            self.average_test_loss.append(self.function_test())
            print('Epoch:{}, train loss:{:.4f}, test loss :{:.4f}'.format(epoch + 1, self.average_train_loss[epoch],
                                                                          self.average_test_loss[epoch]))

        return self.average_train_loss, self.average_test_loss

    def function_test(self):
        """This method tests the autoencoder model on the test data and returns the average test loss."""
        test_loss = []
        for Image, label in self.test_loader:
            imgIn2Autoencoder = torch.clone(Image)
            Image = Image.to(self.device)

            if self.noise:
                imgIn2Autoencoder = imgIn2Autoencoder + self.level_of_noise * np.random.normal(
                    size=imgIn2Autoencoder.shape).astype(
                    "float32")

            imgIn2Autoencoder = imgIn2Autoencoder.to(self.device)
            OutPut = self(imgIn2Autoencoder)
            loss = self.criterion(OutPut, Image)
            test_loss.append(float(loss))

        return np.array(test_loss).mean()

    def printResult(self, col):
        """This code defines the printResult method of the Autoencoder class. This method takes in an integer col and
        visualizes the original, noisy (if noise is enabled), and reconstructed images for a specified number of
        images in the test set. It does this by first loading a batch of test images, adding noise to the images if
        noise is enabled, and passing the images through the autoencoder model to get the reconstructed outputs. The
        original, noisy, and reconstructed images are then plotted in a grid of subplots, with col subplots per row.
        The number of rows in the plot depends on whether noise is enabled: if noise is enabled, there will be three
        rows (one for the original images, one for the noisy images, and one for the reconstructed images),
        while if noise is disabled, there will be only two rows (one for the original and one for the reconstructed
        images). The show method is then called to display the plot. """
        dataiter = iter(self.test_loader)
        Image, label = next(dataiter)
        imgIn2Autoencoder = torch.clone(Image)
        Image = Image.numpy()
        if self.noise:
            imgIn2Autoencoder = imgIn2Autoencoder + self.level_of_noise * np.random.normal(
                size=imgIn2Autoencoder.shape).astype(
                "float32")
        noisy_images = deepcopy(imgIn2Autoencoder)
        imgIn2Autoencoder = imgIn2Autoencoder.to(self.device)
        OutPut = self(imgIn2Autoencoder)
        OutPut = OutPut.view(64, 1, 28, 28)
        OutPut = OutPut.to('cpu')
        OutPut = OutPut.detach().numpy()
        howManyRow = 3 if self.noise else 2
        plt.figure(figsize=(20, 10))
        for i in range(col):
            ax = plt.subplot(howManyRow, col, i + 1)
            plt.imshow(Image[i].squeeze(), cmap='gray')
            ax.set_axis_off()

            if self.noise:
                ax = plt.subplot(howManyRow, col, i + 1 + (howManyRow - 2) * col)
                plt.imshow(noisy_images[i].squeeze(), cmap='gray')
                ax.set_axis_off()

            ax = plt.subplot(howManyRow, col, i + 1 + (howManyRow - 1) * col)
            plt.imshow(OutPut[i].squeeze(), cmap='gray')
            ax.set_axis_off()

        plt.show()

    def print_loss(self):
        """This method plots the training and test losses over the number of epochs."""
        fig, ax = plt.subplots()
        ax.plot(range(self.epochs), self.average_train_loss, label='Training Loss')
        ax.plot(range(self.epochs), self.average_test_loss, label='Testing Loss')
        ax.set_title('Training and Testing Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_xticks(range(self.epochs))
        ax.legend(loc='best')
        plt.show()


def main(noise=True, n_image2print=12):
    """This code defines a main function that creates an instance of the Autoencoder class, trains it on the MNIST
    dataset, and visualizes the original, noisy (if noise is enabled), and reconstructed images for a number of
    images in the test set. It also plots the training and test losses over the number of epochs. """
    _Autoencoder = Autoencoder(noise=noise, level_of_noise=0.4, epochs=20, batchSize=64, learningRate=1e-4)
    _avg_train_loss, _avg_test_loss = _Autoencoder.train()
    _Autoencoder.printResult(n_image2print)
    _Autoencoder.print_loss()


if __name__ == '__main__':
    dic = {'Q1': 'Images Without noise', 'Q2': 'Images With noise'}
    for key, value in dic.items():
        print(f'{key}: {value}')
        if key == 'Q1':
            main(noise=False)
        elif key == 'Q2':
            main()
