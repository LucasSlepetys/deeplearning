import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = (X_train.astype(np.float32) / 255.0)[..., None] # (60000,28,28,1)
X_test  = (X_test.astype(np.float32)  / 255.0)[..., None]  # (10000,28,28,1)
Y_train = np.eye(10)[y_train] #one hot encode of value (m, 10)
Y_test  = np.eye(10)[y_test] #(m, 10)

def printType():

    print(f"X_train: {X_train.shape, X_train.dtype}")
    print("\n")
    print(f"X_test: {X_test.shape, X_test.dtype}")
    print("\n")
    print(f"Y_train: {Y_train.shape, Y_train.dtype}")
    print("\n")
    print(f"Y_train: {Y_train.shape, Y_train.dtype}")
    print("\n")
    print(f"Y_test: {Y_test.shape, Y_test.dtype}")
    print("\n")
    print("Data from mnist dataset")

def loadData():

    return X_train, X_test, Y_train, Y_test

def showOneData():

    i = np.random.randint(0, 5000)
    img = X_train[i]          # (28, 28, 1)
    img = np.squeeze(img)  # -> (28, 28)
    
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
    plt.show()
