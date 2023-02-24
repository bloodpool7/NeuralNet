import numpy as np
import time

def mnist_load_data(input_file = "Data/train-images-idx3-ubyte", label_file = "Data/train-labels-idx1-ubyte"):
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
            input.append(x[index]/255)
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

def mnist_load_test(input_file = "Data/t10k-images-idx3-ubyte", label_file = "Data/t10k-labels-idx1-ubyte"):
    print("Loading test data...")
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
            input.append(x[index]/255)
            index += 1
        inputs.append(input)
    
    print("Finished!\n")
    print("Loading test labels...")

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

def shuffle_data(inputs, labels, batch_size = 1):
    batches = []
    targets = []

    batch = []
    label = []
    for i in range(len(inputs)):
        if (i+1) % batch_size == 0:
            batch.append(np.array(inputs[i]))
            label.append(np.array(labels[i]))
            batches.append(np.array(batch))
            targets.append(np.array(label))
            batch = []
            label = []
        else:
            batch.append(np.array(inputs[i]))
            label.append(np.array(labels[i]))
    
    if (len(batch) > 0):
        batches.append(np.array(batch))
    if (len(label) > 0):
        targets.append(np.array(label))

    return (np.array(batches, dtype = np.ndarray), np.array(targets, dtype = np.ndarray))


if __name__ == "__main__":
    data = [[1, 2, 3], [7 ,8 ,9], [10, 12, 13], [0.3, 0.6, 0.19], [18, 19, 20]]
    labels = [0, 2, 1, 2, 0]
    inputs, labels = mnist_load_data()
    pictures = []
    for i in range(10):
        picture = []
        for j in range(len(inputs[0])):
            picture.append(inputs[i][j])
            if (len(picture) == 28):
                pictures.append(picture)
                picture = []

    pictures = np.array(pictures)
    np.set_printoptions(linewidth = 150)
    for i in range(10):
        for line in pictures[i*28:(i+1)*28]:
            print(line)
    print(labels[:10])