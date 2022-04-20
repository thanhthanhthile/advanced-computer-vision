
import tensorflow as tf
# print(tf.__version__)
import matplotlib.pyplot as plt
import numpy as np


# Data: y = 8x + 2
n_samples = 100
X_train = np.linspace(-5, 5, n_samples)
y_train = 8 * X_train + 2
noise =  4 * np.random.randn(n_samples)
y_train += noise

# plt.scatter(X_train, y_train, color='blue', marker='.', label='Training Data')
# plt.plot(X_train, 8 * X_train + 2, color='orange')
# plt.show()

# Linear Regression

# parameters
learning_rate = 0.01
training_steps = 100
display_step = 2


# Weight and Bias, initialized randomly.
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Linear regression (Wx + b).
def linear_regression(x):
    return W * x + b

# Mean square error.
def mean_square(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Gradient Descent Optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

# Optimization process. 
def run_optimization():
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = linear_regression(X_train)
        loss = mean_square(pred, y_train)

    # Compute gradients.
    gradients = g.gradient(loss, [W, b])
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))


# Run training for the given number of steps.
for step in range(1, training_steps + 1):
    # Run the optimization to update W and b values.
    run_optimization()

    plt.scatter(X_train, y_train, color='blue', marker='.', label='Data')

    if step % display_step == 0:
        pred = linear_regression(X_train)
        loss = mean_square(pred, y_train)
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))

        # Visually
        x_vis = np.array([-5.0, 5.0])
        y_vis = W * x_vis + b
        plt.plot(x_vis, y_vis)
        plt.pause(0.1)
