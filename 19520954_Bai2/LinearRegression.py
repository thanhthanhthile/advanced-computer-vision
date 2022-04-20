import numpy as np
import matplotlib.pyplot as plt


# Phat sinh du lieu
x = np.arange(-5, 5, 0.5)
n_sample = len(x)
noise = np.random.normal(0, 2, n_sample) 
Y = -10*x + 4 + noise
plt.plot(x, Y, 'bo')

# Khoi tao tham so theta, alpha, eps
ones = np.ones((1, n_sample))
X = np.concatenate((ones, [x]))
theta = np.array([[8], [-2]])
alpha = 0.01
eps = 0.001

# Loop
while True:
    nabla = (1.0/n_sample)*np.dot(X, (np.dot(theta.T, X) - Y).T)
    theta = theta - alpha*nabla
    
    # Truc quan hoa
    x_vis = np.array([-5.0, 5.0])
    y_vis = theta[1][0] * x_vis + theta[0][0]
    plt.plot(x_vis, y_vis)
    plt.pause(0.1)

    nabla = (1.0/n_sample)*np.dot(X, (np.dot(theta.T, X) - Y).T)
    if abs(nabla[0][0]) < eps and abs(nabla[1][0]) < eps:
        break 

# Show ket qua
print("Gia tri cua theta: ", theta)
plt.show()

