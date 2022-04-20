import numpy as np
import matplotlib.pyplot as plt

# Phat sinh du lieu
X = np.random.rand(2, 100)
y = np.expand_dims(np.random.randint(2, size=100), axis=0)



# Ham sigmoid, ham loss, dao ham loss

def sigmoid(X, theta):
  z = np.dot(theta.T, X)
  return 1. / (1 + np.exp(-z))

def Loss(y, y_pred):
  return np.mean((-y * np.log(y_pred) + (1-y) * np.log(1-y_pred)))

def der_loss(y_pred, y, X):
  return np.dot(X, (y_pred - y).T)

# Cac tham so
theta = np.random.rand(2, 1)
lr = 0.01
loss = []

# Loop
for epoch in range(100):
  y_pred = sigmoid(X, theta)
  theta = theta - lr * der_loss(y_pred, y ,X)
  loss.append(- Loss(y,y_pred)) 

# Show ket qua

print(f'Loss: {loss[-1]}')
print(f'Theta: {theta[0,0]}')

plt.plot(range(100), loss)
plt.show()