import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self):
        self.l1 = np.random.uniform(0, 1, (2, 3))
        self.l2 = np.random.uniform(0, 1, (3, 3))
        self.output = np.random.uniform(0, 1, (3, 1))
    
    def forward(input):
        # TODO      
        return outpus
    
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))


    def derivative_sigmoid(x):
        return np.multiply(x, 1.0 - x)


    def log_loss():
        # TODO
        pass


    def log_pred(pred_y):
        # TODO
        pass


