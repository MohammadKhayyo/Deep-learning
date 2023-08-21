# Simple RNN Character-level Language Model

Hello there! 👋 Dive into a character-level language model built using a simple recurrent neural network. This code is not only an excellent demonstration of a foundational concept in deep learning but a clear testament to my expertise in designing and implementing RNN architectures. I'm confident this will spark your interest, and I welcome an opportunity to discuss the model in detail.

## 🚀 Features:
1. **Three-layer Neural Network**: Input (one-hot vector), Hidden (RNN layer), and Output (Softmax layer).
2. **Character-level Modeling**: Predicts the next character based on the sequence of previous characters.
3. **Custom Loss Function**: Combines forward and backward passes for model training.
4. **Adaptive Learning**: Uses Adagrad optimization for adaptive learning rates.
5. **Sampling Function**: Given a seed character, predict and generate the next sequence of characters.

## 📦 Requirements:
- `numpy`

## 📈 How It Works:

- **Data Preprocessing**: The code reads a plain text file, creating a vocabulary of unique characters. Each character is mapped to a unique integer and vice versa.

- **Network Initialization**: We initialize the weights of the network. There are three main weight matrices:
  - \(W_{xh}\): Input to Hidden layer
  - \(W_{hh}\): Hidden to Hidden layer (Recurrence!)
  - \(W_{hy}\): Hidden to Output layer

- **Training Loop**: For each batch of data:
  1. Forward Pass: Computes the predicted probabilities for the next character in sequence.
  2. Compute Loss: Uses cross-entropy loss.
  3. Backward Pass: Calculates gradients via backpropagation through time.
  4. Update Weights: Uses Adagrad optimization for weight updates.
  5. Sample: Every 100 iterations, a sample sequence is generated from the model.

- **Adagrad**: Helps in adjusting the learning rate during training, preventing aggressive updates which can destabilize the model.

## 📝 TODO:
- Consider augmenting the model with additional features.
- Explore different optimization techniques beyond Adagrad.
- Extend the RNN architecture to more complex forms like LSTM/GRU for better handling of long sequences.

## 🤖 Running the Code:
Simply execute the Python script. Ensure that you have an `input.txt` file in the same directory.

```bash
$ python rnn_script.py
```

## 🤔 Questions & Feedback:
I'm open to discussions about this model and its potential improvements. If you have any suggestions or would like a deeper dive into the mechanics of this RNN, please feel free to reach out!

---

**📞 Contact**:

Name : Mohammad Khayyo  
Email : mohammadkhayyo@gmail.com  
LinkedIn : linkedin.com/in/mohammadkhayyo
---

**P.S.**: If you enjoyed reading this README as much as I did crafting it, let's discuss the code behind it in our next conversation! 🌟

---

> _"The biggest room in the world is the room for improvement."_ - **Helmut Schmidt**