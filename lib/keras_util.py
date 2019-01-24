""" Shared utility functions for keras interaction. """

import keras
import json
from keras.callbacks import EarlyStopping
from keras.layers import Lambda
import numpy as np
import math
import random

isCallHistory = False
random.seed(1)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def save(self, dir_path, fold_filename, layers_config, alpha, filePrefix):
        file_prefix = '{:s}_history_a={:.4f}_'.format(filePrefix, alpha) + 'H'.join(str(x) for x in np.array(layers_config))
        history_filepath = '{!s}/folds/{!s}_{!s}.csv'.format(dir_path, file_prefix, fold_filename[:-4])

        np.savetxt(history_filepath, self.losses, delimiter=',', header="MSE", comments='')
        print("saved: {!s}".format(history_filepath))


def fitModel(model, input, output, layers_config, alpha, prefix, dir_path, fold_filename):
    nM = input.shape[0]
    rmsprop = keras.optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=rmsprop, loss='mse', metrics=['mse'])
    
    
    earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=100, verbose=1, mode='min')

    if isCallHistory:
        history = LossHistory()
        model.fit(input, output, epochs=20000, batch_size=nM, callbacks=[history, earlystop], verbose=1)
        history.save(dir_path, fold_filename, layers_config, alpha, prefix)
    else:
        model.fit(input, output, epochs=20000, batch_size=nM, callbacks=[earlystop], verbose=0)

def getMLPHyperParameters(min_nodes, max_nodes, min_alpha, max_alpha, num_combinations):
    h_list = []
    
    # hidden_layers=0: 10%
    n0 = math.ceil(num_combinations * 0.1)
    for i in range(0, n0):
        alpha = random.uniform(min_alpha, max_alpha)
        h_list.append([alpha, []])

    # hidden_layers=1: 20%
    n1 = math.ceil(num_combinations * 0.2)
    for i in range(0, n1):
        alpha = random.uniform(min_alpha, max_alpha)
        nodes_l1 = random.randint(min_nodes, max_nodes)
        h_list.append([alpha, [nodes_l1]])

    # hidden_layers=2: 30%
    n2 = math.ceil(num_combinations * 0.3)
    for i in range(0, n2):
        alpha = random.uniform(min_alpha, max_alpha)
        nodes_l1 = random.randint(min_nodes, max_nodes)
        nodes_l2 = random.randint(min_nodes, max_nodes)
        h_list.append([alpha, [nodes_l1, nodes_l2]])

    # hidden_layers=3: 40%
    n3 = num_combinations - len(h_list)
    for i in range(0, n3):
        alpha = random.uniform(min_alpha, max_alpha)
        nodes_l1 = random.randint(min_nodes, max_nodes)
        nodes_l2 = random.randint(min_nodes, max_nodes)
        nodes_l3 = random.randint(min_nodes, max_nodes)
        h_list.append([alpha, [nodes_l1, nodes_l2, nodes_l3]])

    return h_list

def getRnnHyperParameters(max_depth, min_alpha, max_alpha, num_combinations):
    h_list = []
    for i in range(0, num_combinations):
        alpha = random.uniform(min_alpha, max_alpha)
        depth = random.randint(1, max_depth)
        h_list.append([alpha, depth])
    
    return h_list

def getLassoHyperParameters(min_alpha, max_alpha, num_combinations):
    h_list = []
    for i in range(0, num_combinations):
        alpha = random.uniform(min_alpha, max_alpha)
        h_list.append([alpha, []])
    return h_list

def loadMLPHyperParameters(dir_path):
    filename = "{!s}/../../../hparam_kmlp.json".format(dir_path)
    with open(filename, 'r') as fHandle:
        h_list = json.load(fHandle)

    return h_list

def apply_range(model, ge_range_all, output_cols):
    mins = []
    maxs = []
    for col in output_cols:
        mins.append(ge_range_all[col][0])
        maxs.append(ge_range_all[col][1])
    
    mul = np.asarray(maxs) - np.asarray(mins)
    add = np.asarray(mins)

    model.add(Lambda(lambda x: x * mul))
    model.add(Lambda(lambda x: x + add))

    return model
