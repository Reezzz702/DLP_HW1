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

        if 0.1 * i == 0.5:
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

def show_learning_accuracy_curve(epochs, losses, accuracies):
    plt.subplot(1, 2, 1)
    plt.title('Learning curve', fontsize=18)
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.title('Accuracy curve', fontsize=18)
    plt.plot(epochs, accuracies, 'g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.show()


class network():
    def __init__(self, input_size, output_size, h1_size, h2_size):
        self.w1 = np.random.rand(input_size, h1_size)
        self.w2 = np.random.rand(h1_size, h2_size)
        self.w3 = np.random.rand(h2_size, output_size)
    

    def forward(self, x):
        self.x = x
        self.z1 = sigmoid(self.x @ self.w1)
        self.z2 = sigmoid(self.z1 @ self.w2)
        self.z3 = sigmoid(self.z2 @ self.w3)
        self.pred_y = self.z3
        return self.pred_y
    

    def backpropagation(self, y):
        dL_z3 = derivative_MSE(y, self.pred_y)
        dz3_z2w2 = derivative_sigmoid(self.z3)
        dz2_z1w2 = derivative_sigmoid(self.z2) #dz2_z1w2 = derivative_ReLU(self.z2)
        dz1_xw1 = derivative_sigmoid(self.z1)  #dz1_xw1 = derivative_ReLU(self.z1)

        self.dL_w3 = self.z2.T @ (dz3_z2w2 * dL_z3)
        self.dL_w2 = self.z1.T @ (dz2_z1w2  * ((dz3_z2w2 * dL_z3) @ self.w3.T))
        self.dL_w1 = self.x.T @ (dz1_xw1 * ((dz2_z1w2  * ((dz3_z2w2 * dL_z3) @ self.w3.T)) @ self.w2.T))

    def update_weight(self, lr):
        self.w1 = self.w1 - lr * self.dL_w1
        self.w2 = self.w2 - lr * self.dL_w2
        self.w3 = self.w3 - lr * self.dL_w3

    

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)


def derivative_MSE(y, pred_y):
    return -2 * (y - pred_y) / y.shape[0]



def train(args):
    if args.task == "linear":
        x, y = generate_linear()
    else:
        x, y = generate_XOR_easy()

    model = network(2, 1, 8, 4)

    losses = []
    acc_list = []
    for i in range(1, args.epoch + 1):
        pred_y = model.forward(x)
        loss = np.mean((y-pred_y) ** 2)
        losses.append(loss)
        model.backpropagation(y)
        model.update_weight(args.lr)
        
        acc = np.sum(np.where(pred_y > 0.5, 1, 0) == y) / len(y) * 100
        acc_list.append(acc)
        if i % 1000 == 0:
            print(f"epoch {i} loss : {loss} acc(%) : {acc}")
    show_learning_accuracy_curve(range(1, args.epoch + 1), losses, acc_list)
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
