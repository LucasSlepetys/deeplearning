#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Libraries being used:
import numpy as np

#Loading bar
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy


# In[5]:


def relu_derivative(Z):

    return (Z > 0).astype(float)

def sigmoid_derivative(Z):

    S = 1 / (1 + np.exp(-Z))

    return S * (1 - S)

def entropy_loss_function_derivative(AL, Y):

    dAL = - (Y / AL) + ((1 - Y) / (1 - AL))

    return dAL


# In[79]:


class L_layer_NN:

    #Later: add mini-batch
    #add Clipping, but first understand where clipping is required and why and how it fixes it and what it is

    def __init__(self, layer_dims, learning_rate=0.01, lambd = 0, optimizer = "gd", initialization = "he", beta1 = 0.9, beta2 = 0.999, batch_size = 64):

        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.optimizer = optimizer.lower()
        self.initialization = initialization
        self.parameters = self._initialize_parameters()
        self.costs = []
        self.grads = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.epoch = 1
        self.batch_size = batch_size


        self._initialize_optimizer()

    def _initialize_parameters(self):

        parameters = {}

        L = len(self.layer_dims)

        for l in range(1, L):

            #He initialization
            if(self.initialization == "he"): 
                parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2.0 / self.layer_dims[l-1])
                parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

            #random initialization 
            elif(self.initialization == "random"):
                parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
                parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

            #throw error
            else:
                raise ValueError(f'Unknown parameter initialization : "{self.initialization }" | please try: he, random, etc')

        return parameters



    def _linear_forward(self, A, W, b):

        Z = W @ A + b

        #cache used for _linear_backward
        linear_cache = (A, W, b)

        return Z, linear_cache



    def _linear_activation_forward(self, A_prev, W, b, activation):

        Z, linear_cache = self._linear_forward(A_prev, W, b)
        #cache used for activation _linear_activation_backward
        activation_cache = Z

        if activation == "relu":
            A = np.maximum(0, Z)
        elif activation == "sigmoid":
            A = 1 / (1 + np.exp(-Z))
        elif activation == "linear": 
            A = Z
        else:
            raise ValueError(f"Unknown activation: {activation}") 

        cache = (linear_cache, activation_cache)

        return A, cache



    def _forward(self, X):

        A = X
        L = len(self.layer_dims) - 1
        caches = []

        for l in range(1, L):

            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]

            A, cache = self._linear_activation_forward(A, W, b, "relu")
            caches.append(cache)

        W = self.parameters["W" + str(L)]
        b = self.parameters["b" + str(L)]

        AL, cache = self._linear_activation_forward(A, W, b, "sigmoid")
        caches.append(cache)

        return AL, caches

    def _compute_cost(self, AL, Y):

        m = Y.shape[1]
        #Cost-entropy cost function:
        eps = 1e-8
        AL_clipped = np.clip(AL, eps, 1 - eps)
        cost = -(1 / m) * (Y @ np.log(AL_clipped).T + (1 - Y) @ np.log(1 - AL_clipped).T)

        #L2 regularization
        L2_term = 0 
        for key in self.parameters:
            if key.startswith("W"):
                L2_term += np.sum(np.square(self.parameters[key])) 

        #add l2 reg into cost
        cost += (self.lambd / (2 * m)) * L2_term

        assert cost.shape == (1, 1)

        return np.squeeze(cost)

    def _linear_backward(self, dZ, cache):

        A_prev, W, b = cache

        m = A_prev.shape[1]

        dW = (dZ @ A_prev.T) / m 

        db = np.sum(dZ, axis = 1, keepdims=True) / m

        dA_prev = W.T @ dZ

        return dA_prev, dW, db

    def _linear_activation_backward(self, dA, cache, activation):

        linear_cache, Z = cache

        if activation == "relu":

            dZ = dA * relu_derivative(Z)

        elif activation == "sigmoid":

            dZ = dA * sigmoid_derivative(Z)

        elif activation == "linear":

            dZ = dA

        else: raise ValueError(f"Unknown activation {activation}")

        dA_prev, dW, db = self._linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def _backward(self, AL, Y, caches):

        L = len(self.layer_dims) - 1

        dAL = entropy_loss_function_derivative(AL, Y)
        dA = dAL

        current_cache = caches[L - 1]

        dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(dAL, current_cache, "sigmoid")

        self.grads["dA" + str(L-1)] = dA_prev_temp
        self.grads["dW" + str(L)] = dW_temp
        self.grads["db" + str(L)] = db_temp

        for l in reversed(range(L - 1)):

            current_cache = caches[l]

            dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(dA_prev_temp, current_cache, "relu" )

            self.grads["dA" + str(l)] = dA_prev_temp
            self.grads["dW" + str(l + 1)] = dW_temp
            self.grads["db" + str(l + 1)] = db_temp

    def _initialize_optimizer(self):

        if self.optimizer == "momentum":

            self.velocities = {}
            L = len(self.layer_dims) - 1
            for l in range(L):

                self.velocities["vdW" + str(l+1)] = np.zeros_like(self.parameters["W" + str(l+1)])
                self.velocities["vdb" + str(l+1)] = np.zeros_like(self.parameters["b" + str(l+1)])

        if self.optimizer == "rmsprop":

            self.squares = {}
            L = len(self.layer_dims) - 1
            for l in range(L):

                self.squares["sdW" + str(l+1)] = np.zeros_like(self.parameters["W" + str(l+1)])
                self.squares["sdb" + str(l+1)] = np.zeros_like(self.parameters["b" + str(l+1)])

        if self.optimizer == "adam":

            self.velocities = {}
            self.squares = {}

            L = len(self.layer_dims) - 1

            for l in range(L):

                self.velocities["vdW" + str(l+1)] = np.zeros_like(self.parameters["W" + str(l+1)])
                self.velocities["vdb" + str(l+1)] = np.zeros_like(self.parameters["b" + str(l+1)])

                self.squares["sdW" + str(l+1)] = np.zeros_like(self.parameters["W" + str(l+1)])
                self.squares["sdb" + str(l+1)] = np.zeros_like(self.parameters["b" + str(l+1)])


    def _update_gd(self):

        L = len(self.layer_dims) - 1
        m = self.grads["dA" + str(L - 1)].shape[1]

        for l in range(L):

            _dW = self.grads["dW" + str(l+1)] + (self.lambd / m) * self.parameters["W" + str(l + 1)] # -> dW with L2 reg
            self.parameters["W" + str(l+1)] -= self.learning_rate * _dW
            self.parameters["b" + str(l+1)] -= self.learning_rate * self.grads["db" + str(l+1)]


    def _update_momentum(self):

        L = len(self.layer_dims) - 1
        m = self.grads["dA" + str(L - 1)].shape[1]

        for l in range(L):

            _dW = self.grads["dW" + str(l+1)] + (self.lambd / m) * self.parameters["W" + str(l + 1)] # -> dW with L2 reg

            #update velocities:
            self.velocities["vdW" + str(l+1)] = self.beta1 * self.velocities["vdW" + str(l+1)] + (1-self.beta1) * _dW
            self.velocities["vdb" + str(l+1)] = self.beta1 * self.velocities["vdb" + str(l+1)] + (1-self.beta1) * self.grads["db" + str(l+1)]

            #update parameters
            self.parameters["W" + str(l+1)] -= self.learning_rate * self.velocities["vdW" + str(l+1)]
            self.parameters["b" + str(l+1)] -= self.learning_rate * self.velocities["vdb" + str(l+1)]

    def _update_rmsprop(self):

        L = len(self.layer_dims) - 1
        m = self.grads["dA" + str(L - 1)].shape[1]

        for l in range(L):

            _dW = self.grads["dW" + str(l+1)] # -> dW without L2 reg

            #update squares:
            self.squares["sdW" + str(l+1)] = self.beta2 * self.squares["sdW" + str(l+1)] + (1-self.beta2) * _dW**2
            self.squares["sdb" + str(l+1)] = self.beta2 * self.squares["sdb" + str(l+1)] + (1-self.beta2) * self.grads["db" + str(l+1)]**2

            #update parameters
            eps = 1e-8
            self.parameters["W" + str(l+1)] -= self.learning_rate * _dW / (np.sqrt( self.squares["sdW" + str(l+1)]) + eps) 
            self.parameters["b" + str(l+1)] -= self.learning_rate * self.grads["db" + str(l+1)] / (np.sqrt( self.squares["sdb" + str(l+1)]) + eps) 


    def _update_adam(self):

        L = len(self.layer_dims) - 1
        m = self.grads["dA" + str(L - 1)].shape[1]

        for l in range(L):

            _dW = self.grads["dW" + str(l+1)] # -> dW without L2 reg

            #update velocities:
            self.velocities["vdW" + str(l+1)] = self.beta1 * self.velocities["vdW" + str(l+1)] + (1-self.beta1) * _dW
            self.velocities["vdb" + str(l+1)] = self.beta1 * self.velocities["vdb" + str(l+1)] + (1-self.beta1) * self.grads["db" + str(l+1)]

            #update squares:
            self.squares["sdW" + str(l+1)] = self.beta2 * self.squares["sdW" + str(l+1)] + (1-self.beta2) * _dW**2
            self.squares["sdb" + str(l+1)] = self.beta2 * self.squares["sdb" + str(l+1)] + (1-self.beta2) * self.grads["db" + str(l+1)]**2

            #bias correction for velocities:
            _vdW_corrected = self.velocities["vdW" + str(l+1)] / (1 - self.beta1**self.epoch)
            _vdb_corrected = self.velocities["vdb" + str(l+1)] / (1 - self.beta1**self.epoch)

            #bias correction for squares:
            _sdW_corrected = self.squares["sdW" + str(l+1)] / (1 - self.beta2**self.epoch)
            _sdb_corrected = self.squares["sdb" + str(l+1)] / (1 - self.beta2**self.epoch)

            #update parameters:
            eps = 1e-8
            self.parameters["W" + str(l+1)] -= self.learning_rate * _vdW_corrected / ( np.sqrt(_sdW_corrected) + eps ) 
            self.parameters["b" + str(l+1)] -= self.learning_rate * _vdb_corrected / ( np.sqrt(_sdb_corrected) + eps )




    def _update_parameters(self):

        if self.optimizer == "gd": self._update_gd()
        elif self.optimizer == "momentum": self._update_momentum()
        elif self.optimizer == "rmsprop": self._update_rmsprop()
        elif self.optimizer == "adam": self._update_adam()
        else: raise ValueError(f"Unknown optimizer: {self.optimizer}")


    def _grad_check(self):
        #add other values to be gotten by function
        pass

    def accuracy(self, X, y):

        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)

        return (accuracy * 100)

    def _suffle_into_mini_batches(self, X, y):

        m = X.shape[1]

        #suffle data
        permutation = np.random.permutation(X.shape[1])
        X_shuffled = X[:, permutation]
        y_shuffled = y[:, permutation]

        mini_batches = []

        for i in range(0, m, self.batch_size):

            X_batch = X_shuffled[:, i : i + self.batch_size]
            y_batch = y_shuffled[:, i : i + self.batch_size]

            batch = np.vstack((X_batch, y_batch)) #shape (n_x + n_y, batch_size)
            mini_batches.append(batch)

        return mini_batches




    def train(self, X, y, epochs = 3000, print_cost=True):

        n = X.shape[0]

        progress = tqdm(range(epochs), desc="Training", leave=True)

        for i in progress:

            mini_batches = self._suffle_into_mini_batches(X, y) #array with mini batchs, each batch shape: (n_x + n_y, batch_size) or (n_x + 1, batch_size)
            epoch_cost = []
            for batch in mini_batches:

                batch_X = batch[:n, :]
                batch_y = batch[n: , :]
                #cjange the computation of cost to compute every #epochs or however design decision i want
                AL, caches = self._forward(batch_X)
                cost = self._compute_cost(AL, batch_y)

                self._backward(AL, batch_y, caches)
                self._update_parameters()
                epoch_cost.append(cost)

            mean_cost = np.mean(epoch_cost)

            if i % 10 == 0:
                self.costs.append((i, mean_cost))
                if(print_cost): progress.set_postfix({ "Mean cost": f"{mean_cost}" })

            self.epoch += 1

    def predict(self, X):

        AL, _ = self._forward(X)

        return (AL > 0.5).astype(int)

    def plot_cost(self):

        iters, values = zip(*self.costs)
        plt.plot(iters, values)
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.title("Cost vs Epoch")
        plt.grid(True)
        plt.show()



# In[80]:


X = np.random.randn(2, 5) * 10
y = np.array([[1, 1, 1, 0, 0]])

print(np.vstack((X, y)))
print("\n\n")

NN = L_layer_NN([X.shape[1], 5, 1], batch_size = 2)
mini_batches = NN._suffle_into_mini_batches(X, y)

for batch in mini_batches:
    print(batch[:X.shape[0], :])
    print(batch[X.shape[0]:, :])
    print("\n")




# In[ ]:




