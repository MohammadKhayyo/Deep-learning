# Name : Mohammad Khayyo
# ID : 211558895
import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # --------------------------------------------------------------------
    def forward(self, X):
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        y1 = np.dot(X, W1) + b1
        h1 = np.where(y1 > 0, y1, 0)
        y2 = np.dot(h1, W2) + b2

        return (y2, h1, y1)

    def predict(self, X):
        y_pred = None
        (y2, h1, y1) = self.forward(X)
        y_pred = np.argmax(y2, axis=1)

        return y_pred

    # ----------------------------------------------------------------------------
    def computeLoss(self, NetOut, y):
        N = len(y)

        exp_scores = np.exp(NetOut)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

        # select all corresponding log likelihood score for each class
        # N being the number of sample and take the mean of it
        corect_logprobs = -np.log(probs[range(N), y])
        loss = np.sum(corect_logprobs) / N  # np.mean()
        return loss

    def backPropagation(self, NetOut, h1, y1, X, y, reg):
        grads = {}

        # need to backprop loss = np.sum(-score_correct_classes + np.log(total_score))/N
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Derivative of Cross Entropy Loss with Softmax
        exp_scores = np.exp(NetOut)
        dy2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # TODO
        dy2[range(N), y] -= 1
        dy2 /= N

        dh1 = dy2.dot(W2.T)

        # gradient TODO
        dy1 = dh1 * (y1 >= 0)

        # gradient TODO
        dW1 = X.T.dot(dy1)
        dW2 = h1.T.dot(dy2)

        db1 = np.sum(dy1, axis=0)
        db2 = np.sum(dy2, axis=0)

        # Regularization
        dW1 += reg * W1
        dW2 += reg * W2

        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['b2'] = db2
        return grads

    # ----------------------------------------------------------------------------
    def lossAndGrad(self, X, y=None, reg=0.0):

        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # Compute the forward pass
        (NetOut, h1, y1) = self.forward(X)

        # If the targets are not given then jump out, we're done
        if y is None:
            return NetOut

        # Compute the loss
        loss = self.computeLoss(NetOut, y)

        # TODO
        reg_loss = reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss += reg_loss

        grads = self.backPropagation(NetOut, h1, y1, X, y, reg)

        return loss, grads

    # ----------------------------------------------------------------------------
    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        N, D = X.shape

        for it in range(num_iters):

            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx]
            y_batch = y[idx]

            loss, grads = self.lossAndGrad(X_batch, y=y_batch, reg=reg)

            loss_history.append(loss)

            self.params['W1'] += -learning_rate * grads['W1']
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b1'] += -learning_rate * grads['b1']
            self.params['b2'] += -learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }
