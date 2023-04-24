#
#
#  Simple RNN model
#
#

## this is a 3 layers neuron network.
## input layer: one hot vector, dim: vocab * 1
## hidden layer: RNN, hidden vector: hidden size * 1
## output layer: Softmax, vocab * 1, the probabilities distribution of each character

import numpy as np

data = open('input.txt', 'r').read()  # simple plain text file

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperWseters
hidden_size = 100  # hidden layer
seq_length = 25  # number of steps to unroll the RNN
learning_rate = 1e-1

# model

# TODO-1: fill in model size.
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias


# ------------------------------------------------------------------------------
#  network + loss
#
#  Given
#     inputs, targets -- list of integers.
#     hprev           --  Hx1 array of initial hidden state
#     returns the loss, gradients on model Wseters, and last hidden state
# ------------------------------------------------------------------------------
def lossFun(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}

    ## record hidden state
    hs[-1] = np.copy(hprev)
    loss = 0

    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1

        ## hidden state, using Wxh, Whh,  xs[t] and previous hidden state hs[t-1]
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)

        ## next chars
        ys[t] = np.dot(Why, hs[t]) + by

        ## probabilities for next chars (use softmax)
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        ## softmax (cross-entropy loss)
        loss += -np.log(ps[t][targets[t], 0])

    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # backprop into y

        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)

    for dWs in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dWs, -5, 5, out=dWs)  # clip to mitigate exploding gradients

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


# ------------------------------------------------------------------------------
#  given a hidden RNN state, and a input char id, predict the coming n chars
# ------------------------------------------------------------------------------
def sample(h, seed_ix, n):
    ## a one-hot vector
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1

    ixes = []
    for t in range(n):
        ## hidden state, using Wxh, Wxh,  xs[t] and previous hidden state hs[t-1]
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)

        ## Output
        y = np.dot(Why, h) + by
        ## softmax

        ## probabilities for next chars, softmax
        p = np.exp(y) / np.sum(np.exp(y))
        ## sample next chars according to probability distribution
        ix = np.random.choice(range(vocab_size), p=p.ravel())

        ## update input x
        ## use the new sampled result as last input, then predict next char again.
        x = np.zeros((vocab_size, 1))
        x[ix] = 1

        ixes.append(ix)

    return ixes


n, p = 0, 0  # n is iteration counter, p is character index in the data sequence
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
## main loop
while True:

    if p + seq_length + 1 >= len(data) or n == 0:
        # reset RNN dx_gradory
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        # go from start of data
        p = 0

    inputs = [char_to_ix[ch] for ch in data[p: p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1: p + seq_length + 1]]

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('---- sample -----')
        print('----\n %s \n----' % (txt,))

    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)

    ## Adagrad + momentum
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n % 100 == 0:
        print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

    # parameter update
    for Ws, dWs, dx_grad in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
        dx_grad += dWs * dWs
        Ws += -learning_rate * dWs / np.sqrt(dx_grad + 1e-8)  # adagrad update

    p += seq_length  # move data pointer
    n += 1  # iteration counter
