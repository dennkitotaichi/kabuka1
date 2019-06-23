import numpy as np
class Perceptron(object):
    def __init__(self, learn, itr):
        """
        :param learn: learning rate
        :type learn: float
        :param itr: number of iterations
        :type itr: integer
        """
        self.learn = learn
        self.itr = itr
    def train(self, x, y):
        """
        Train the weights with data set x for outputs y
        :param x: training data features
        :type x: array (matrix) of n-samples, n-features
        :param y: training data outputs
        :type y: array of n-samples
        :return: weights (w) and errors for each iteration
        """
        self.w = np.zeros(1 + x.shape[1])
        self.error = []
        for _ in range(self.itr):
            errors = 0
            for xi, target in zip(x, y):
                update = self.learn*(target - self.predict(xi))
                self.w[1:] += update*xi
                self.w[0] += update
                errors += int(update != 0)
            self.error.append(errors)
        return self
    def predict(self, x):
        """
        :param x: input vector of features
        :type x: ndarray
        :return: int 1 or -1
        """
        self.output = np.dot(x, self.w[1:]) + self.w[0]
        return np.where(self.output >= 0, 1, -1)

    def feedforward(self, x):
        """
        Predict the output given the inputs
        :param x: input vector of features
        :type x: ndarray
        :return: All activation values and x values.
        """
        self.z0 = np.dot(self.w0, x)
        self.output1 = self.sigmoid(self.z0)
        self.z1 = np.matmul(self.w1, self.output1)
        self.output2 = self.sigmoid(self.z1)
        self.z2 = np.matmul(self.w2, self.output2)
        self.output3 = self.sigmoid(self.z2)
        return self.z0, self.output1, self.z1, self.output2, self.z2, self.output3



