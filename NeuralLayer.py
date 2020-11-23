import numpy as np


class NeuralLayer:

    def __init__(self, num_neurons):
        self.neurons = num_neurons
        self.a = np.ones(num_neurons)

        # any initializations
        self.delta = np.zeros(num_neurons)
        self.dl_dw = 0
        self.momentum = 0
        self.accumulated = 0

        self.momentum_b = 0
        self.accumulated_b = 0

        self.dl_db = 0
        self.weights = 0
        self.bias = 0
        self.drop_out_mask = 1

    def set_weights(self, w, b):
        self.weights = w
        self.bias = b

    def set_drop_out_mask(self, mask):
        self.drop_out_mask = mask

    @staticmethod
    def compute(w, x, b):
        z = np.dot(x, w) + b

        return z

    @staticmethod
    def activate_relu(z):
        return np.maximum(0, z)

    @staticmethod
    def diff_relu(z):
        return 1.0 * (z > 0)

    @staticmethod
    def activate_soft_max(z):
        exp = np.exp(z)

        return exp/np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def diff_tanh(a):
        return 1.0 - np.tanh(a)**2

    @staticmethod
    def activate_tanh(z):
        return  np.tanh(z)
