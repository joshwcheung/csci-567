"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.
"""

import json
import numpy as np
import sys
import dnn_misc
import os
import argparse


def data_loader_mnist(dataset):
    # This function reads the MNIST data and separate it into train, val, and test set
    with open(dataset, 'r') as f:
        data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    Ytrain = train_set[1]
    Xvalid = valid_set[0]
    Yvalid = valid_set[1]
    Xtest = test_set[0]
    Ytest = test_set[1]

    return np.array(Xtrain).reshape(-1, 1, 28, 28), np.array(Ytrain), np.array(Xvalid).reshape(-1, 1, 28, 28),\
           np.array(Yvalid), np.array(Xtest).reshape(-1, 1, 28, 28), np.array(Ytest)


def predict_label(f):
    # This is a function to determine the predicted label given scores
    if f.shape[1] == 1:
        return (f > 0).astype(float)
    else:
        return np.argmax(f, axis=1).astype(float).reshape((f.shape[0], -1))


class DataSplit:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N, self.d, _, _ = self.X.shape

    def get_example(self, idx):
        batchX = np.zeros((len(idx), self.X.shape[1], self.X.shape[2], self.X.shape[3]))
        batchY = np.zeros((len(idx), 1))
        for i in range(len(idx)):
            batchX[i] = self.X[idx[i]]
            batchY[i, :] = self.Y[idx[i]]
        return batchX, batchY


def main(main_params):

    ### set the random seed ###
    np.random.seed(int(main_params['random_seed']))

    ### data processing ###
    Xtrain, Ytrain, Xval, Yval , _, _ = data_loader_mnist(dataset = 'mnist_subset.json')
    N_train, d, _, _ = Xtrain.shape
    N_val, _, _, _ = Xval.shape

    trainSet = DataSplit(Xtrain, Ytrain)
    valSet = DataSplit(Xval, Yval)

    ### building/defining CNN ###
    """
    In this script, we are going to build a CNN for a 10-class classification problem on MNIST.
    The network structure is input --> convolution --> relu --> max pooling --> flatten --> dropout --> linear --> softmax_cross_entropy loss
    the hidden_layer size (num_L1) is 4900
    the output_layer size (num_L2) is 10
    """
    model = dict()
    num_L1 = 4900
    num_L2 = 10

    # experimental setup
    num_epoch = int(main_params['num_epoch'])
    minibatch_size = int(main_params['minibatch_size'])

    # optimization setting: _alpha for momentum, _lambda for weight decay
    _learning_rate = float(main_params['learning_rate'])
    _step = 10
    _alpha = float(main_params['alpha'])
    _lambda = float(main_params['lambda'])
    _dropout_rate = float(main_params['dropout_rate'])

    # create objects (modules) from the module classes
    model['C1'] = dnn_misc.conv_layer(num_input = d, num_output = 25, filter_len = 5, stride = 1)
    model['nonlinear1'] = dnn_misc.relu()
    model['M1'] = dnn_misc.max_pool(max_len = 2, stride = 2)
    model['F1'] = dnn_misc.flatten_layer()
    model['drop1'] = dnn_misc.dropout(r = _dropout_rate)
    model['L1'] = dnn_misc.linear_layer(input_D = num_L1, output_D = num_L2)
    model['loss'] = dnn_misc.softmax_cross_entropy()

    # create variables for momentum
    if _alpha > 0.0:
        momentum = dnn_misc.add_momentum(model)
    else:
        momentum = None

    train_acc_record = []
    val_acc_record = []

    ### run training and validation ###
    for t in range(num_epoch):
        print('At epoch ' + str(t + 1))
        if (t % _step == 0) and (t != 0):
            _learning_rate = _learning_rate * 0.1

        idx_order = np.random.permutation(N_train)

        train_acc = 0.0
        train_loss = 0.0
        train_count = 0

        val_acc = 0.0
        val_count = 0

        for i in range(int(np.floor(N_train / minibatch_size))):

            # get a mini-batch of data
            x, y = trainSet.get_example(idx_order[i * minibatch_size : (i + 1) * minibatch_size])

            ### forward ###
            c1 = model['C1'].forward(x)
            h1 = model['nonlinear1'].forward(c1)
            m1 = model['M1'].forward(h1)
            f1 = model['F1'].forward(m1)
            d1 = model['drop1'].forward(f1, is_train = True)
            a1 = model['L1'].forward(d1)
            loss = model['loss'].forward(a1, y)

            ### backward ###
            grad_a1 = model['loss'].backward(a1, y)
            grad_d1 = model['L1'].backward(d1, grad_a1)
            grad_f1 = model['drop1'].backward(f1, grad_d1)
            grad_m1 = model['F1'].backward(m1, grad_f1)
            grad_h1 = model['M1'].backward(h1, grad_m1)
            grad_c1 = model['nonlinear1'].backward(c1, grad_h1)
            grad_x = model['C1'].backward(x, grad_c1)

            ### gradient_update ###
            for module_name, module in model.items():

                # check if a module has learnable parameters
                if hasattr(module, 'params'):
                    for key, _ in module.params.items():
                        g = module.gradient[key] + _lambda * module.params[key]

                        if _alpha > 0.0:
                            momentum[module_name + '_' + key] = _alpha * momentum[module_name + '_' + key] - _learning_rate * g
                            module.params[key] += momentum[module_name + '_' + key]

                        else:
                            module.params[key] -= _learning_rate * g

        ### Computing training accuracy and obj ###
        for i in range(int(np.floor(N_train / minibatch_size))):

            x, y = trainSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))

            ### forward ###
            c1 = model['C1'].forward(x)
            h1 = model['nonlinear1'].forward(c1)
            m1 = model['M1'].forward(h1)
            f1 = model['F1'].forward(m1)
            d1 = model['drop1'].forward(f1, is_train = False)
            a1 = model['L1'].forward(d1)
            loss = model['loss'].forward(a1, y)
            train_loss += len(y) * loss
            train_acc += np.sum(predict_label(a1) == y)
            train_count += len(y)

        train_loss = train_loss / train_count
        train_acc = train_acc / train_count
        train_acc_record.append(train_acc)

        print('Training loss at epoch ' + str(t + 1) + ' is ' + str(train_loss))
        print('Training accuracy at epoch ' + str(t + 1) + ' is ' + str(train_acc))

        ### Computing validation accuracy ###
        for i in range(int(np.floor(N_val / minibatch_size))):

            x, y = valSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))

            ### forward ###
            c1 = model['C1'].forward(x)
            h1 = model['nonlinear1'].forward(c1)
            m1 = model['M1'].forward(h1)
            f1 = model['F1'].forward(m1)
            d1 = model['drop1'].forward(f1, is_train = False)
            a1 = model['L1'].forward(d1)
            val_acc += np.sum(predict_label(a1) == y)
            val_count += len(y)

        val_acc = val_acc / val_count
        val_acc_record.append(val_acc)

        print('Validation accuracy at epoch ' + str(t + 1) + ' is ' + str(val_acc))

    # save file
    json.dump({'train': train_acc_record, 'val': val_acc_record},
              open('CNN_lr' + str(main_params['learning_rate']) +
                   '_m' + str(main_params['alpha']) +
                   '_w' + str(main_params['lambda']) +
                   '_d' + str(main_params['dropout_rate']) +
                   '.json', 'w'))

    print('Finish running!')
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=2)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--alpha', default=0.0)
    parser.add_argument('--lambda', default=0.0)
    parser.add_argument('--dropout_rate', default=0.5)
    parser.add_argument('--num_epoch', default=30)
    parser.add_argument('--minibatch_size', default=5)
    args = parser.parse_args()
    main_params = vars(args)
    # print ('parsed input parameters:')
    # print (json.dumps(main_params, indent = 2))
    main(main_params)