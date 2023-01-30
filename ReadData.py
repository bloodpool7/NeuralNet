import numpy as np
import time

f = open("train-images-idx3-ubyte", "rb")
x = f.read()
n = int.from_bytes(x[3:4], "big") - 1

length = int.from_bytes(x[4:8], "big")
d = []
for i in range(n):
    d.append(int.from_bytes(x[8 + i * 4: 12 + i * 4], "big"))

index = 8 + n*4
inputs = []
for i in range(length):
    input = []
    for j in range(d[0] * d[1]):
        input.append(x[index])
        index += 1
    inputs.append(input)

pictures = []
for i in range(10):
    picture = []
    for j in range(len(inputs[0])):
        picture.append(inputs[i][j])
        if (len(picture) == 28):
            pictures.append(picture)
            picture = []

print(len(pictures))

pictures = np.array(pictures)
np.set_printoptions(linewidth = 150)
for i in range(10):
    for line in pictures[i*28:(i+1)*28]:
        print(line)