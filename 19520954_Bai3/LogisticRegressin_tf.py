from random import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Data
means = [[2, 5], [6, 8]]
cov = [[1, 0], [0, 1]]

NUMBER_OF_SAMPLE_PER_CLASS = 500
n_sample = NUMBER_OF_SAMPLE_PER_CLASS * 2

positive = np.random.multivariate_normal(means[0], cov, NUMBER_OF_SAMPLE_PER_CLASS)
negative = np.random.multivariate_normal(means[1], cov, NUMBER_OF_SAMPLE_PER_CLASS)

X = np.concatenate((positive, negative), axis = 0).T

original_label = np.asarray([0]*NUMBER_OF_SAMPLE_PER_CLASS + [1]*NUMBER_OF_SAMPLE_PER_CLASS).T
X = np.transpose(X)
Y = original_label
print('Shape X: ', X.shape)
print('Shape Y: ', Y.shape)

neg_idx = [i for i in range(len(original_label)) if original_label[i] == 0 ]
pos_idx = [i for i in range(len(original_label)) if original_label[i] == 1]

negative = X[neg_idx]
positive = X[pos_idx]

plt.scatter(negative[:,0], negative[:,1], color='red')
plt.scatter(positive[:,0], positive[:,1], color='blue')
plt.legend(['0', '1'])
plt.show()

# Logistic Regression

# parameters
learning_rate = 0.01
training_steps = 100
display_step = 2

# Weight of shape [784, 10], the 28*28 image features, and total number of classes.
W = tf.Variable(np.random.randn(), name="weight")
# Bias of shape [10], the total number of classes.
b = tf.Variable(np.random.randn(), name="bias")

# Logistic regression (Wx + b).
def logistic_regression(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred),1))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

# Optimization process. 
def run_optimization():
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = logistic_regression(X)
        loss = cross_entropy(pred, Y)

    # Compute gradients.
    gradients = g.gradient(loss, [W, b])
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))

# Run training for the given number of steps.
for step in range(1, training_steps + 1):
    # Run the optimization to update W and b values.
    run_optimization()
    
    if step % display_step == 0:
        pred = logistic_regression(X)
        loss = cross_entropy(pred, Y)
        acc = accuracy(pred, Y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

        # # Visually
        # x_vis = np.array([-5.0, 5.0])
        # y_vis = W * x_vis + b
        # plt.plot(x_vis, y_vis)
        # plt.pause(0.1)