import numpy as np
import matplotlib.pyplot as plt

from neural_net  import TwoLayerNet

from data_utils import get_CIFAR10_data



# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
X_train.astype("float64")



input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)

##
#  Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
 

plt.subplot(2, 1, 2)
l1 = plt.plot(stats['train_acc_history'], label='train')
l2 = plt.plot(stats['val_acc_history'], label='val')
#plt.legend(handles=[l1, l2])
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()


from vis_utils import visualize_grid


def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

learning_rates = [1e-3, 1e-4]
regularization_strengths = [0.5, 1]
hidden_size = [50, 500]
results = {}
best_net = None # store the best model into this
best_val = -1
best_size = -1
acc = lambda v, v_pred: np.mean(v == v_pred)

for lr in learning_rates:
    for k in regularization_strengths:
        for h in hidden_size:
            net = TwoLayerNet(input_size, h, num_classes)
            net.train(X_train, y_train, X_val, y_val,
                      num_iters=1500, batch_size=200,
                      learning_rate=lr, learning_rate_decay=0.95,
                      reg=k, verbose=True)
            y_train_pred = net.predict(X_train)
            y_val_pred = net.predict(X_val)
            trainbest = acc(y_train, y_train_pred)
            valbest = acc(y_val, y_val_pred)
            if valbest > best_val:
                best_val = valbest
                best_net = net
                best_size = h

            results[(lr, k)] = (trainbest, valbest)

show_net_weights(best_net)


# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e , %e , train accuracy: %f val accuracy: %f' % (
        lr, reg, best_size , train_accuracy, val_accuracy))

print ('best validation accuracy achieved during cross-validation: %f' % best_val)