import numpy as np

class SSP:
    def __init__(self, n, c):
        self.c = c
        self.n = n
        self.weights = np.random.randint(1, 180, self.n)

    def greedy(self):
        pass

