import numpy as np
import time

def mnist_load_data(input_file = "Data/train-images-idx3-ubyte", label_file = "train-labels-idx1-ubyte"):
    print("Loading data...")
    f = open(input_file, "rb")
    x = f.read()
    n = int.from_bytes(x[3:4], "big") - 1
    length = int.from_bytes(x[4:8], "big")
    d = []
    for i in range(n):
        d.append(int.from_bytes(x[8 + i * 4: 12 + i * 4], "big"))
    
    d = np.array(d)
    sample_length = np.prod(d)

    index = 8 + n*4
    inputs = []
    for i in range(length):
        input = []
        for j in range(sample_length):
            input.append(x[index])
            index += 1
        inputs.append(input)
    
    print("Finished!\n")
    print("Loading labels...")

    f = open(label_file, "rb")
    x = f.read()
    n = int.from_bytes(x[3:4], "big") - 1
    length = int.from_bytes(x[4:8], "big")

    index = 8 + n*4
    labels = []
    for i in range(length):
        labels.append(x[index])
        index += 1
    print("Finished!\n")
    inputs = np.array(inputs)
    labels = np.array(labels)
    return inputs, labels

def iris_load_data(input_file = "Data/iris.data"):
    f = open(input_file, "r")
    inputs = []
    labels = []
    while True:
        line = f.readline()
        if line and len(line) != 1:
            line = line.split(",")
            line[-1] = line[-1][:-1]
            inputs.append(list(map(float, line[:-1])))
            labels.append(0 if line[-1] == "Iris-setosa" else (1 if line[-1] == "Iris-versicolor" else 2))
        else:
            break
    return np.array(inputs), np.array(labels)


if __name__ == "__main__":
    inputs, labels = iris_load_data()
    print(inputs)
    # inputs, labels = mnist_load_data()
    # pictures = []
    # for i in range(10):
    #     picture = []
    #     for j in range(len(inputs[0])):
    #         picture.append(inputs[i][j])
    #         if (len(picture) == 28):
    #             pictures.append(picture)
    #             picture = []

    # pictures = np.array(pictures)
    # np.set_printoptions(linewidth = 150)
    # for i in range(10):
    #     for line in pictures[i*28:(i+1)*28]:
    #         print(line)
    # print(labels[:10])