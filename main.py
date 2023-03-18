import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append(pt[0], pt[1])
        # distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    lables = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*1])
        lables.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*1])
        lables.append(1)

    return np.array(inputs), np.array(lables).reshape(21, 1)


def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.show()

class network():
    def __init__(self, input_size, output_size, h1_size, h2_size, lr):
        self.l1 = np.random.rand((input_size, h1_size))
        self.l2 = np.random.rand((h1_size, h2_size))
        self.output = np.random.rand((h2_size, output_size))
        self.lr = lr
    

    def forward(self, x):
        self.x = x
        self.z1 = self.sigmoid(self.x @ self.l1)
        self.z2 = self.sigmoid(self.z1 @ self.l2)
        self.pred_y = self.sigmoid(self.z2 @ self.output)
        return self.pred_y
    

    def backpropagation(self, y, pred_y):
        dy = self.derivative_MSE(y, pred_y)
        dz3 = self.derivative_sigmoid(self.pred_y)
        dz2 = self.derivative_sigmoid(self.z2)
        dz1 = self.derivative_sigmoid(self.z1)

        self.d_l3 = self.z2.T @ (dz3 * dy)
        self.d_l2 = self.z1.T @ (dz2 * ((dz3 * dy) @ self.output.T))
        self.d_l1 = self.x.T @ (dz1 * ((dz2 * ((dz3 * dy) @ self.output.T)) @ self.l2.T))


    def update(self):
        self.l1 = self.l1 - self.lr * self.d_l1
        self.l2 = self.l2 - self.lr * self.d_l2
        self.l3 = self.l3 - self.lr * self.d_l3
    

    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))


    def derivative_sigmoid(x):
        return np.multiply(x, 1.0 - x)


    def MSE(y, pred_y):
        return np.mean(np.linalg.norm(y-pred_y))
    

    def derivative_MSE(y, pred_y):
        return -2 * (y - pred_y) / len(y)


    def log_loss():
        # TODO
        pass


    def log_pred(pred_y):
        # TODO
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="linear", help="select task: linear, xor")
    parser.add_argument("--lr", type=float, default=0.1, help="set learning rate")
    parser.add_argument("--activation", type=b)