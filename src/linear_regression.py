''''
Fill in your names:
group member 1: Tom Freudenmann
group member 2: Jonathan 

'''

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

# Load the raw MNIST
X_train: np.ndarray = datasets.MNIST('./data', train=True, download=True).data.numpy()
y_train: np.ndarray = datasets.MNIST('./data', train=True, download=True).targets.numpy()

X_test: np.ndarray = datasets.MNIST('./data', train=False, download=True).data.numpy()
y_test: np.ndarray = datasets.MNIST('./data', train=False, download=True).targets.numpy()

# split eval data from train data:
eval_data_size = 10000
train_data_size = 50000
test_data_size = 10000

X_eval: np.ndarray = X_train[0:eval_data_size, :, :]
y_eval: np.ndarray = y_train[0:eval_data_size]
X_train: np.ndarray = X_train[eval_data_size:, :, :]
y_train: np.ndarray = y_train[eval_data_size:]
# As a sanity check, we print out the size of the training and test data.
def print_shapes():
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Evaluation data shape: ', X_eval.shape)
    print('Evaluation labels shape: ', y_eval.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
print_shapes()

# Reshape the image data into rows
# IMPORTANT NOTE:
#In the lecture the so-called design matrix is defined to be the matrix
#with rows as the data points (in this exercise the flattened images).
#However, in the assignment sheet the design matrix is defined to be the
#matrix with columns as the data points

# Datatype float allows you to subtract images (is otherwise uint8)
X_train = np.reshape(X_train, (X_train.shape[0], -1)).astype('float') # Does 28x28 -> 784
X_eval = np.reshape(X_eval, (X_eval.shape[0], -1)).astype('float')
X_test = np.reshape(X_test, (X_test.shape[0], -1)).astype('float')
print("x shapes:")
print(X_train.shape, X_eval.shape, X_test.shape)
# normalize train data from range 0 to 255 to range 0 to 1
X_train = X_train / 255
X_eval = X_eval / 255
X_test = X_test / 255


# transform to y to one hot encoded vectors:
# each row is one y vector
def make_one_hot(v: np.ndarray):
    """
    :param v: vector of the length of the dataset containing class labels from 0 to 9
    :return: a matrix of dim(lenght dataset,10), where the index of the corresponding label is set to one.
    """
    v_one_hot = np.zeros((v.shape[0], 10))
    for i in range(v.shape[0]):
        v_one_hot[i, v[i]] = 1    
    return v_one_hot


y_train = make_one_hot(y_train)
y_eval = make_one_hot(y_eval)
y_test = make_one_hot(y_test)
print("y shapes:")
print(y_train.shape, y_eval.shape, y_test.shape)

# TODO for task d adapt the following parameters
batch_size = 100
epochs = 20
learning_rate = 0.001

# usually one would use a random weight initialization, but for reproduceable results we use fixed weights
# Don't change these parameters
W = np.ones((784, 10)) * 0.01
b = np.ones((10)) * 0.01


def get_next_batch(iteraton, batch_size, data, label):
    X = data[iteraton * batch_size:(iteraton + 1) * batch_size, :]
    y = label[iteraton * batch_size:(iteraton + 1) * batch_size, :]
    return X, y


def softmax(x):
    """
    :param x The input dim(batch_size, 10)
    :return Result of the softmax dim(batch_size, 10)
    """
    # Substitution for causing no numerical instabilities
    z = x - np.max(x, axis=-1, keepdims=True)
    
    # Implementation which might cause numerical instabilities
    result = np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)
    
    return result


def get_loss(y_hat: np.ndarray, y: np.ndarray):
    """
    :param y_hat is the output from fully connected layer dim(batch_size,10)
    :param y is labels dim(batch_size,10)
    :return Loss dim(1)
    """
    epsilon = 1e-10
    loss = -np.sum(y * np.log(y_hat + epsilon))
    return loss


def get_accuracy(y_hat: np.ndarray, y: np.ndarray):
    """
    the accuracy for one image is one if the maximum of y_hat has the same index as the 1 in y
    :param y_hat:  dim(batch_size,10)
    :param y: dim(batch_size,10)
    :return: mean accuracy dim(1)
    """
    acc = 0
    for row_y, row_y_hat in zip(y, y_hat):
        acc += 1 if row_y.argmax() == row_y_hat.argmax() else 0
    return acc / y.shape[0]


def do_network_inference(x: np.ndarray):  # over whole batch
    """
    :param x: Input dim(batchsize,784)
    :return: Inference output dim(batchsize,10)
    """
    z = np.dot(x, W) + b
    y_hat = softmax(z)
    return y_hat


def get_delta_weights(y_hat, y, x_batch):
    """
    :param y_hat: Inference result dim(batchsize,10)
    :param y: Ground truth dim(batchsize,10)
    :param x_batch: Input data dim(batchsize,784)
    :return: Delta weights dim(748,10)
    """
    delta_weights = x_batch.T@(y_hat - y)
    return delta_weights


def get_delta_biases(y_hat, y):
    """
    :param y_hat: Inference result dim(batchsize,10)
    :param y: Ground truth dim(batchsize,10)
    :return: Delta biases dim(10)
    """
    delta_biases = y_hat - y
    delta_biases = np.mean(delta_biases, axis=0)
    return delta_biases


def do_parameter_update(delta_w, delta_b, W, b):
    """
    :param delta_w: dim(748,10)
    :param delta_b: dim(10)
    :param W: dim(748,10)
    :param b: dim(10)
    """
    W = W - learning_rate * delta_w
    b = b - learning_rate * delta_b
    return W, b

# do training and evaluation
mean_eval_losses = []
mean_train_losses = []
mean_eval_accs = []
mean_train_accs = []

for epoch in range(epochs):
    # training
    mean_train_loss_per_epoch = 0
    mean_train_acc_per_epoch = 0
    for i in range(train_data_size // batch_size):
        x, y = get_next_batch(i, batch_size, X_train, y_train)
        y_hat = do_network_inference(x)
        train_loss = get_loss(y_hat, y)
        train_accuracy = get_accuracy(y_hat, y)
        delta_w = get_delta_weights(y_hat, y, x)
        delta_b = get_delta_biases(y_hat, y)

        W, b = do_parameter_update(delta_w, delta_b, W, b)
        # print(delta_w, delta_b)
        mean_train_loss_per_epoch += train_loss
        mean_train_acc_per_epoch += train_accuracy
        # print("epoch: {0:d} \t iteration {1:d} \t train loss: {2:f}".format(epoch, i,train_loss))

    mean_train_loss_per_epoch = mean_train_loss_per_epoch / (
        (train_data_size // batch_size))
    mean_train_acc_per_epoch = mean_train_acc_per_epoch / (
        (train_data_size // batch_size))
    print("epoch:{0:d} \t mean train loss: {1:f} \t mean train acc: {2:f}".
          format(epoch, mean_train_loss_per_epoch, mean_train_acc_per_epoch))

    # evaluation:
    # training
    mean_eval_loss_per_epoch = 0
    mean_eval_acc_per_epoch = 0
    for i in range(eval_data_size // batch_size):
        x, y = get_next_batch(i, batch_size, X_eval, y_eval)
        y_hat = do_network_inference(x)
        train_loss = get_loss(y_hat, y)
        train_accuracy = get_accuracy(y_hat, y)
        delta_w = get_delta_weights(y_hat, y, x)
        delta_b = get_delta_biases(y_hat, y)

        W, b = do_parameter_update(delta_w, delta_b, W, b)
        mean_eval_loss_per_epoch += train_loss
        mean_eval_acc_per_epoch += train_accuracy
        # print("epoch: {0:d} \t iteration {1:d} \t train loss: {2:f}".format(epoch, i,train_loss))

    mean_eval_loss_per_epoch = mean_eval_loss_per_epoch / (
        (eval_data_size // batch_size))
    mean_eval_acc_per_epoch = mean_eval_acc_per_epoch / (
        (eval_data_size // batch_size))
    
    print(
        "epoch:{0:d} \t mean eval loss: {1:f} \t mean eval acc: {2:f}".format(
            epoch, mean_eval_loss_per_epoch, mean_eval_acc_per_epoch))

    mean_eval_losses.append(mean_eval_loss_per_epoch)
    mean_train_losses.append(mean_train_loss_per_epoch)
    mean_eval_accs.append(mean_eval_acc_per_epoch)
    mean_train_accs.append(mean_train_acc_per_epoch)

# testing
mean_test_loss_per_epoch = 0
mean_test_acc_per_epoch = 0
for i in range(test_data_size // batch_size):
    x, y = get_next_batch(i, batch_size, X_test, y_test)
    y_hat = do_network_inference(x)
    train_loss = get_loss(y_hat, y)
    train_accuracy = get_accuracy(y_hat, y)
    delta_w = get_delta_weights(y_hat, y, x)
    delta_b = get_delta_biases(y_hat, y)

    W, b = do_parameter_update(delta_w, delta_b, W, b)
    mean_test_loss_per_epoch += train_loss
    mean_test_acc_per_epoch += train_accuracy 
    # print("epoch: {0:d} \t iteration {1:d} \t train loss: {2:f}".format(epoch, i,train_loss))

mean_test_loss_per_epoch = mean_test_loss_per_epoch / (
    (test_data_size // batch_size))
mean_test_acc_per_epoch = mean_test_acc_per_epoch / (
    (test_data_size // batch_size))

print("final test loss: {0:f} \t final test acc: {1:f}".format(
    mean_test_loss_per_epoch, mean_test_acc_per_epoch))

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
ax1 = axs[0]
ax2 = axs[1]

ax1.plot(range(epochs), mean_train_losses, "r", label="train loss")
ax1.plot(range(epochs), mean_eval_losses, "b", label="eval loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.legend()

ax2.plot(range(epochs), mean_train_accs, "r", label="train acc")
ax2.plot(range(epochs), mean_eval_accs, "b", label="eval acc")
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.legend()
plt.show()
