# CIFAR-10 Data Utilities & Two Layer Neural Network

Hello and welcome! If you're reading this, you're likely interested in diving deep into how to utilize and structure data utilities for the CIFAR-10 dataset and to create a basic two-layer neural network. 

## Features 🌟
1. **Cross Compatibility**: This codebase is compatible with both Python 2 and Python 3.
2. **Optimized Data Processing**: CIFAR-10 data is loaded, reshaped, normalized, and prepared for classifiers in one smooth operation.
3. **Neural Network from Scratch**: Train a two-layer neural network without the bells and whistles of major deep learning frameworks. Get to know the inner workings!
4. **Visualizations**: Understand training dynamics by plotting loss and accuracy graphs.

## Files 📂

### 1. `data_utils.py`
*Handles the CIFAR-10 data loading and preprocessing.*

- **Functions**:
  - `load_pickle(f)`: A Python 2 and 3 compatible function to load pickled data.
  - `load_CIFAR_batch(filename)`: Load a single batch of CIFAR.
  - `load_CIFAR10(ROOT)`: Load all CIFAR batches.
  - `get_CIFAR10_data(...)`: Load and preprocess CIFAR-10 data for classifiers.
  - `get_CIFAR10_data_SVM_softmax(...)`: Special preprocessing for linear classifier.

### 2. `Main_two_layer_net.py`
*Main driver code to invoke data utility functions, train the neural network, and visualize results.*

- **Highlights**:
  - Loads CIFAR-10 data.
  - Defines network parameters and initializes a two-layer neural network.
  - Trains the network and displays training dynamics.
  - Cross-validation for hyperparameter search.

### 3. `neural_net.py`
*The brain of our operations. Defines the architecture, forward and backward passes, loss computations, and training mechanics of our two-layer neural network.*

- **Class**:
  - `TwoLayerNet`: Defines the neural network, forward & backward prop, loss computation, and training loop.
  
## How to Use ⚙️

1. Ensure you have the CIFAR-10 dataset in the `'cifar-10-batches-py'` directory.
2. Run the `Main_two_layer_net.py` to train the two-layer neural network.
3. Analyze the visualizations to see the training dynamics.
4. Dive into `neural_net.py` to understand and potentially modify the architecture.

## About the Author 👤

- **Name**: Mohammad Khayyo

Feel free to explore the code, understand the architecture, and even enhance the network! If you're a recruiter or potential employer, I'd be thrilled to discuss the intricacies of this project and my motivation behind specific design choices. Looking forward to your feedback or potential opportunities to collaborate!

**Note**: Always ensure you have the right dependencies installed and are using a compatible Python version.

## Feedback and Support 💬

For any queries or feedback
🔗 [LinkedIn](https://www.linkedin.com/in/mohammadkhayyo/) | 📧 Email: mohammadkhayyo@gmail.com

Happy coding!
