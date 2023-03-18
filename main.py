import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        # distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []
    
    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


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
        self.l1 = np.random.rand(input_size, h1_size)
        self.l2 = np.random.rand(h1_size, h2_size)
        self.l3 = np.random.rand(h2_size, output_size)
        self.lr = lr
    

    def forward(self, x):
        self.x = x
        self.z1 = self.sigmoid(self.x @ self.l1)
        self.z2 = self.sigmoid(self.z1 @ self.l2)
        self.pred_y = self.sigmoid(self.z2 @ self.l3)
        return self.pred_y
    

    def backward(self, y, pred_y):
        dy = self.derivative_MSE(y, pred_y)
        dz3 = self.derivative_sigmoid(self.pred_y)
        dz2 = self.derivative_sigmoid(self.z2)
        dz1 = self.derivative_sigmoid(self.z1)

        d_l3 = self.z2.T @ (dz3 * dy)
        d_l2 = self.z1.T @ (dz2 * ((dz3 * dy) @ self.l3.T))
        d_l1 = self.x.T @ (dz1 * ((dz2 * ((dz3 * dy) @ self.l3.T)) @ self.l2.T))

        self.l1 = self.l1 - self.lr * d_l1
        self.l2 = self.l2 - self.lr * d_l2
        self.l3 = self.l3 - self.lr * d_l3

    
    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def derivative_sigmoid(x):
        return np.multiply(x, 1.0 - x)

    @staticmethod
    def MSE(y, pred_y):
        return np.mean(np.linalg.norm(y-pred_y))
    
    @staticmethod
    def derivative_MSE(y, pred_y):
        return -2 * (y - pred_y) / len(y)



def train(args):
    if args.task == "linear":
        x, y = generate_linear()
    else:
        x, y = generate_XOR_easy()

    model = network(2, 1, 8, 4, args.lr)

    losses = []
    acc_list = []
    for i in range(args.epoch):
        pred_y = model.forward(x)
        loss = model.MSE(y, pred_y)
        losses.append(loss)
        model.backward(y, pred_y)
        
        acc = np.sum(np.where(pred_y > 0.5, 1, 0) == y) / len(y) * 100
        acc_list.append(acc)
        print(f"epoch {i+1} loss : {loss} acc(%) : {acc}")

    return model


def test(args, model):
    if args.task == "linear":
        x, y = generate_linear()
    else:
        x, y = generate_XOR_easy()

    pred_y = model.forward(x)
    print(f"pred_y:\n{pred_y}")
    pred_y = np.where(pred_y > 0.5, 1, 0)
    acc = np.sum(pred_y == y) / len(y) * 100
    print(20*"=")
    print(f"Accuracy(%): {acc}")
    show_result(x, y, pred_y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="linear", help="select task: linear, xor")
    parser.add_argument("--lr", type=float, default=0.1, help="set learning rate")
    parser.add_argument("-A", "--activation_function", type=str, default="sigmoid", help="select activation function: sigmoid, ReLU, None")
    parser.add_argument("--epoch", type=int, default=20000, help="number of training epoch")

    args = parser.parse_args() 
    model = train(args)
    test(args, model)
