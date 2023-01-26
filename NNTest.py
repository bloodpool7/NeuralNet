import numpy as np
import matplotlib.pyplot as plt

soft_output = np.array([[0.7, 0.1, 0.2],
                        [0.4, 0.3, 0.3]])

dvalues = np.array([[1, 2, 3],
                    [3, 2, 1]])

dinputs = np.empty_like(dvalues)

for i, (singout, singdvalue) in enumerate(zip(soft_output, dvalues)):
    singout = np.reshape(singout, (1,-1))
    jacobian = np.diagflat(singout) - np.dot(singout.T, singout)
    dinputs[i] = np.dot(jacobian, singdvalue)


jacobians = []
for row in soft_output:
    row = np.reshape(np.array([row]), (1, len(soft_output[0])))
    jacobians.append(np.dot(row.T, row))

jacobians = np.array(jacobians)

# #forward layer of 3 neurons

# target = np.array([[1, 0, 0],
#                    [0, 0, 1],
#                    [1, 0, 0],
#                    [0, 0, 1],
#                    [0, 1, 0],
#                    [1, 0, 0],
#                    [0, 1, 0],
#                    [0, 1, 0],
#                    [0, 0, 1]])

# inputs = np.array([[0.4, 0.3, 0.8, 0.1],
#                    [0.7, 0.4, -0.2, 0.3],
#                    [0.8, 0.4, -0.22, 0.4],
#                    [0.5, 0.3, -0.4, 0.6],
#                    [0.1, -0.3, 0.2, 0.7],
#                    [0.4, 0.9, 0.9, -0.3],
#                    [0.2, 0.7, 0.3, 0.2],
#                    [0.1, -0.6, -0.2, 0.3],
#                    [0.1, 0.2, -0.5, 0.98]])
# weights = np.array([[-0.4, -0.7, 0.9, 1.1], # neuron 1
#                    [-0.7, -0.6, 0.8, 0.4], # neuron 2
#                    [0.9, 0.7, 0.7, 0.1]]).T  # neuron 3


# bias = [[0.2, 0.3, -0.4]]

# z = np.dot(inputs, weights) + bias
# output = np.maximum(0, z)

# drelu = np.ones(z.shape)


# drelu[z <= 0] = 0

# #backward
# y = np.array([])
# n = 100
# for i in range(n):
#     drelu = np.ones(z.shape)
#     drelu[z <= 0] = 0
#     dvalue = np.multiply((2 * output - 2 * target)/len(output), drelu)

#     dweights = np.dot(inputs.T, dvalue)
#     dinputs = np.dot(dvalue, weights.T)
#     dbias = np.sum(dvalue, axis = 0, keepdims = True)

#     weights -= 0.5 * dweights
#     bias -= 0.5 * dbias

#     z = np.dot(inputs, weights) + bias
#     output = np.maximum(0, z)
#     y = np.append(y, np.mean((output - target) ** 2))


# print(output)
# cost = (output - target) ** 2
# print(np.mean(cost))

# x = range(n)

# fig, ax = plt.subplots()
# ax.plot(x, y)
# plt.show()

