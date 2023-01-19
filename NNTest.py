import numpy as np
import matplotlib.pyplot as plt

#forward layer of 3 neurons

target = np.array([[1, 0, 0],
                   [0, 0, 1]])

inputs = np.array([[0.3, 0.2, 1.7, 0.5],
                   [0.7, 0.4, -0.2, 0.3]])
weights = np.array([[-0.4, -0.7, 0.9, 1.1], # neuron 1
                   [-0.7, -0.6, 0.8, 0.4], # neuron 2
                   [0.9, 0.7, 0.7, 0.1]]).T  # neuron 3


bias = [[0.2, 0.3, -0.4]]

z = np.dot(inputs, weights) + bias
output = np.maximum(0, z)

drelu = np.ones(z.shape)


drelu[z <= 0] = 0

#backward
y = np.array([])
n = 10
for i in range(n):
    drelu = np.ones(z.shape)
    drelu[z <= 0] = 0
    dvalue = np.multiply(2 * output - 2 * target, drelu)

    dweights = np.dot(inputs.T, dvalue)
    dinputs = np.dot(dvalue, weights.T)
    dbias = np.sum(dvalue, axis = 0, keepdims = True)

    weights -= 0.1 * dweights
    bias -= 0.1 * dbias

    z = np.dot(inputs, weights) + bias
    output = np.maximum(0, z)
    y = np.append(y, np.mean((output - target) ** 2))


print(output)
cost = (output - target) ** 2

x = range(n)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()

