import numpy as np

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

print(inputs[0:2])
