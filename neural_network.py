import numpy as np
import sys
import numpy as np
import pandas as pd
from sklearn import model_selection
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.models import Sequential

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.preprocessing import minmax_scale
import string
import re
from collections import Counter
import csv
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.models import Model



#import tensorflow_datasets as tfds
#tfds.disable_progress_bar()



import tensorflow as tf



class RnnLSTM:


    def __init__( self , trainX, testX, trainY, testY, embeddingSize,model, h1 = 40, h2 = 18):

        print(trainX)
        #self.testy = np.zeros()
        testX = np.asarray(testX)
        testY = np.asarray(testY)
        self.trainX = trainX
        #self.trainY = trainY
        np.random.seed(1)
        x1, m, x_t = trainX.shape
        m2, x2 = testX.shape
        self.testy = np.zeros((m2,1))
        self.trainY = np.zeros((m, 1))
        print(self.testy.shape)
        for i in range (0,m2):
            self.testy[i,0] = testY[i]
        for i in range (0,m):
            self.trainY[i,0] = trainY[i]
        #print ("test")
        #print (self.testy)
        self.pred = np.zeros((trainY.shape))
        #self.da = np.zeros((embeddingSize, m))
        n_x = n_a = embeddingSize
        n_a = 1
        self.da = np.random.randn(n_a, m, x_t)
        #print (trainX.shape)
        self.a0 = np.zeros((n_a, m))
        Wf = np.random.randn(n_a,n_x + n_a)
        bf = np.random.randn(n_a, 1)
        Wi = np.random.randn(n_a,n_x + n_a)
        bi = np.random.randn(n_a, 1)
        Wo = np.random.randn(n_a,n_x + n_a)
        bo = np.random.randn(n_a, 1)
        Wc = np.random.randn(n_a,n_x + n_a)
        bc = np.random.randn(n_a, 1)
        Wy = np.random.randn(1, n_a)
        by = np.random.randn(1, 1)
        self.model = model
        parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc,
                      "by": by}
        self.params = parameters

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def lstm_cell_forward(self, xt, a_past, c_past):


        # Retrieve params from "params"
        weightForget = self.params["Wf"]
        biasForget = self.params["bf"]
        weightUpdate = self.params["Wi"]
        biasUpdate = self.params["bi"]
        weightC = self.params["Wc"]
        biasC = self.params["bc"]
        weightOutput = self.params["Wo"]
        biasOutput = self.params["bo"]
        weightY = self.params["Wy"]
        biasY = self.params["by"]

        # Retrieve dimensions from shapes of xt and Wy
        n_x, m = xt.shape
        n_y, n_a = weightY.shape

        # Concatenate a_past and xt
        concat = np.zeros((n_a + n_x, m))
        concat[: n_a, :] = a_past
        concat[n_a:, :] = xt

        # Compute values for ft, it, cct, c_next, ot, a_next
        ft = self.sigmoid(np.matmul(weightForget, concat) + biasForget)
        it = self.sigmoid(np.matmul(weightUpdate, concat) + biasUpdate)
        cct = np.tanh(np.matmul(weightC, concat) + biasC)
        c_next = (ft * c_past) + (it * cct)
        ot = self.sigmoid(np.matmul(weightOutput, concat) + biasOutput)
        a_next = ot * np.tanh(c_next)

        # Compute prediction of the LSTM cell
        yt_pred = self.softmax(np.matmul(weightY, a_next) + biasY)

        # store values needed for backward propagation in cache
        cache = (a_next, c_next, a_past, c_past, ft, it, cct, ot, xt, self.params)

        return a_next, c_next, yt_pred, cache


    def lstm_cell_backward(self, da_next, dc_next, cache):


        # Retrieve information from "cache"
        (a_next, c_next, a_past, c_past, ft, it, cct, ot, xt, params) = cache

        # Retrieve dimensions from xt's and a_next's shape
        n_x, m = xt.shape
        n_a, m = a_next.shape

        # Compute gates related derivatives
        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
        dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
        dft = (dc_next * c_past + ot * (1 - np.square(np.tanh(c_next))) * c_past * da_next) * ft * (1 - ft)

        concat = np.concatenate((a_past, xt), axis=0)

        # Compute params related derivatives.
        dWf = np.dot(dft, concat.T)
        dWi = np.dot(dit, concat.T)
        dWc = np.dot(dcct, concat.T)
        dWo = np.dot(dot, concat.T)
        dbf = np.sum(dft, axis=1, keepdims=True)
        dbi = np.sum(dit, axis=1, keepdims=True)
        dbc = np.sum(dcct, axis=1, keepdims=True)
        dbo = np.sum(dot, axis=1, keepdims=True)

        # Compute derivatives w.r.t pastious hidden state, pastious memory state and input. Use equations (15)-(17). (≈3 lines)
        da_past = np.dot(params['Wf'][:, :n_a].T, dft) + np.dot(params['Wi'][:, :n_a].T, dit) + np.dot(
            params['Wc'][:, :n_a].T, dcct) + np.dot(params['Wo'][:, :n_a].T, dot)
        dc_past = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
        dxt = np.dot(params['Wf'][:, n_a:].T, dft) + np.dot(params['Wi'][:, n_a:].T, dit) + np.dot(
            params['Wc'][:, n_a:].T, dcct) + np.dot(params['Wo'][:, n_a:].T, dot)

        # Save gradients in dictionary
        gradients = {"dxt": dxt, "da_past": da_past, "dc_past": dc_past, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                     "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

        return gradients

    def forward_pass(self , data):


        # Initialize "caches", which will track the list of all the caches
        caches = []

        # Retrieve dimensions from shapes of x and params['Wy']
        n_x, m, T_x = self.trainX.shape
        n_y, n_a = self.params["Wy"].shape

        # initialize "a", "c" and "y" with zeros
        a = np.zeros((n_a, m, T_x))
        c = a
        y = np.zeros((n_y, m, T_x))

        # Initialize a_next and c_next
        a_next = self.a0
        c_next = np.zeros(a_next.shape)

        # loop over all time-steps

        for t in range(T_x):
            #print (a_next.shape)
            # Update next hidden state, next memory state, compute the prediction, get the cache
            a_next, c_next, yt, cache = self.lstm_cell_forward(self.trainX[:,:, t], a_next, c_next)
            # Save the value of the new "next" hidden state in a
            a[:, :, t] = a_next
            # Save the value of the prediction in y
            #y[:, :, t] = yt
            # Save the value of the next cell state
            c[:, :, t] = c_next
            # Append the cache into caches
            caches.append(cache)
            #print (yt)
            self.pred = yt

        # store values needed for backward propagation in cache
        caches = (caches, self.trainX)
        self.pred = self.model.predict(data)
        return a, y, c, caches, self.pred


    def backward_pass(self, da, caches):


        # Retrieve values from the first cache (t=1) of caches.
        (caches, x) = caches
        (a1, c1, a0, c0, f1, i1, cc1, o1, x1, params) = caches[0]

        # Retrieve dimensions from da's and x1's shapes
        n_a, m, T_x = da.shape
        n_x, m = x1.shape

        # initialize the gradients with the right sizes
        dx = np.zeros((n_x, m, T_x))
        da0 = np.zeros((n_a, m))
        da_pastt = np.zeros(da0.shape)
        dc_pastt = np.zeros(da0.shape)
        dWf = np.zeros((n_a, n_a + n_x))
        dWi = np.zeros(dWf.shape)
        dWc = np.zeros(dWf.shape)
        dWo = np.zeros(dWf.shape)
        dbf = np.zeros((n_a, 1))
        dbi = np.zeros(dbf.shape)
        dbc = np.zeros(dbf.shape)
        dbo = np.zeros(dbf.shape)

        # loop back over the whole sequence
        for t in reversed(range(T_x)):
            # Compute all gradients using lstm_cell_backward
            gradients = self.lstm_cell_backward(da_pastt, dc_pastt, caches[t])
            da_pastt = gradients["da_past"]
            dc_pastt = gradients["dc_past"]
            # Store or add the gradient to the params' pastious step's gradient
            dx[:, :, t] = gradients["dxt"]
            dWf += gradients["dWf"]
            dWi += gradients["dWi"]
            dWc += gradients["dWc"]
            dWo += gradients["dWo"]
            dbf += gradients["dbf"]
            dbi += gradients["dbi"]
            dbc += gradients["dbc"]
            dbo += gradients["dbo"]
        # Set the first activation's gradient to the backpropagated gradient da_past.
        da0 = gradients["da_past"]

        # Store the gradients in a python dictionary
        gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                     "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

        return gradients





    def initialize_adam(self):

        #L = len(parameters) // 2  # number of layers in the neural networks
        v = {}
        s = {}

        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        #for l in range(L):
        v["dWf"] = np.zeros(self.params["Wf"].shape)
        v["dbf"] = np.zeros(self.params["bf"].shape)
        v["dWi"] = np.zeros(self.params["Wi"].shape)
        v["dbi"] = np.zeros(self.params["bi"].shape)
        v["dWc"] = np.zeros(self.params["Wc"].shape)
        v["dbc"] = np.zeros(self.params["bc"].shape)
        v["dWo"] = np.zeros(self.params["Wo"].shape)
        v["dbo"] = np.zeros(self.params["bo"].shape)
        v["dWy"] = np.zeros(self.params["Wy"].shape)
        v["dby"] = np.zeros(self.params["by"].shape)

        s["dWf"] = np.zeros(self.params["Wf"].shape)
        s["dbf"] = np.zeros(self.params["bf"].shape)
        s["dWi"] = np.zeros(self.params["Wi"].shape)
        s["dbi"] = np.zeros(self.params["bi"].shape)
        s["dWc"] = np.zeros(self.params["Wc"].shape)
        s["dbc"] = np.zeros(self.params["bc"].shape)
        s["dWo"] = np.zeros(self.params["Wo"].shape)
        s["dbo"] = np.zeros(self.params["bo"].shape)
        s["dWy"] = np.zeros(self.params["Wy"].shape)
        s["dby"] = np.zeros(self.params["by"].shape)

        return v, s


    def updateWeights(self, grads, v, s, t, learning_rate=0.01,
                                    beta1=0.9, beta2=0.999, epsilon=1e-8):


        #L = len(parameters) // 2  # number of layers in the neural networks
        v_corrected = {}  # Initializing first moment estimate, python dictionary
        s_corrected = {}  # Initializing second moment estimate, python dictionary

        # Perform Adam update on all parameters
        #for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dWf"] = beta1 * v["dWf"] + (1 - beta1) * grads["dWf"]
        v["dbf"] = beta1 * v["dbf"] + (1 - beta1) * grads["dbf"]
        v["dWi"] = beta1 * v["dWi"] + (1 - beta1) * grads["dWi"]
        v["dbi"] = beta1 * v["dbi"] + (1 - beta1) * grads["dbi"]
        v["dWc"] = beta1 * v["dWc"] + (1 - beta1) * grads["dWc"]
        v["dbc"] = beta1 * v["dbc"] + (1 - beta1) * grads["dbc"]
        v["dWo"] = beta1 * v["dWo"] + (1 - beta1) * grads["dWo"]
        v["dbo"] = beta1 * v["dbo"] + (1 - beta1) * grads["dbo"]
        #v["dWy"] = beta1 * v["dWy"] + (1 - beta1) * grads["dWy"]
        #v["dby"] = beta1 * v["dby"] + (1 - beta1) * grads["dby"]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dWf"] = v["dWf"] / (1 - beta1 ** t)
        v_corrected["dbf"] = v["dbf"] / (1 - beta1 ** t)
        v_corrected["dWi"] = v["dWi"] / (1 - beta1 ** t)
        v_corrected["dbi"] = v["dbi"] / (1 - beta1 ** t)
        v_corrected["dWc"] = v["dWc"] / (1 - beta1 ** t)
        v_corrected["dbc"] = v["dbc"] / (1 - beta1 ** t)
        v_corrected["dWo"] = v["dWo"] / (1 - beta1 ** t)
        v_corrected["dbo"] = v["dbo"] / (1 - beta1 ** t)
        #v_corrected["dWy"] = v["dWy"] / (1 - beta1 ** t)
        #v_corrected["dby"] = v["dby"] / (1 - beta1 ** t)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dWf"] = beta2 * s["dWf"] + (1 - beta2) * (grads["dWf"] ** 2)
        s["dbf"] = beta2 * s["dbf"] + (1 - beta2) * (grads["dbf"] ** 2)
        s["dWi"] = beta2 * s["dWi"] + (1 - beta2) * (grads["dWi"] ** 2)
        s["dbi"] = beta2 * s["dbi"] + (1 - beta2) * (grads["dbi"] ** 2)
        s["dWc"] = beta2 * s["dWc"] + (1 - beta2) * (grads["dWc"] ** 2)
        s["dbc"] = beta2 * s["dbc"] + (1 - beta2) * (grads["dbc"] ** 2)
        s["dWo"] = beta2 * s["dWo"] + (1 - beta2) * (grads["dWo"] ** 2)
        s["dbo"] = beta2 * s["dbo"] + (1 - beta2) * (grads["dbo"] ** 2)
        #s["dWy"] = beta2 * s["dWy"] + (1 - beta2) * (grads["dWy"] ** 2)
        #s["dby"] = beta2 * s["dby"] + (1 - beta2) * (grads["dby"] ** 2)


        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dWf"] = s["dWf"] / (1 - beta2 ** t)
        s_corrected["dbf"] = s["dbf"] / (1 - beta2 ** t)
        s_corrected["dWi"] = s["dWi"] / (1 - beta2 ** t)
        s_corrected["dbi"] = s["dbi"] / (1 - beta2 ** t)
        s_corrected["dWc"] = s["dWc"] / (1 - beta2 ** t)
        s_corrected["dbc"] = s["dbc"] / (1 - beta2 ** t)
        s_corrected["dWo"] = s["dWo"] / (1 - beta2 ** t)
        s_corrected["dbo"] = s["dbo"] / (1 - beta2 ** t)
        #s_corrected["dWy"] = s["dWy"] / (1 - beta2 ** t)
        #s_corrected["dby"] = s["dby"] / (1 - beta2 ** t)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        self.params["Wf"] = self.params["Wf"] - learning_rate * v_corrected["dWf"] / np.sqrt(s_corrected["dWf"] + epsilon)
        self.params["bf"] = self.params["bf"] - learning_rate * v_corrected["dbf"] / np.sqrt(s_corrected["dbf"] + epsilon)
        self.params["Wi"] = self.params["Wi"] - learning_rate * v_corrected["dWi"] / np.sqrt(s_corrected["dWi"] + epsilon)
        self.params["bi"] = self.params["bi"] - learning_rate * v_corrected["dbi"] / np.sqrt(s_corrected["dbi"] + epsilon)
        self.params["Wc"] = self.params["Wc"] - learning_rate * v_corrected["dWc"] / np.sqrt(s_corrected["dWc"] + epsilon)
        self.params["bc"] = self.params["bc"] - learning_rate * v_corrected["dbc"] / np.sqrt(s_corrected["dbc"] + epsilon)
        self.params["Wo"] = self.params["Wo"] - learning_rate * v_corrected["dWo"] / np.sqrt(s_corrected["dWo"] + epsilon)
        self.params["bo"] = self.params["bo"] - learning_rate * v_corrected["dbo"] / np.sqrt(s_corrected["dbo"] + epsilon)
        #self.params["dWy"] = self.params["Wy"] - learning_rate * v_corrected["dWy"] / np.sqrt(s_corrected["dWy"] + epsilon)
        #self.params["dby"] = self.params["by"] - learning_rate * v_corrected["dby"] / np.sqrt(s_corrected["dby"] + epsilon)


        return v, s


    def train(self, data, max_iterations = 10, learning_rate = 0.0005 ):
        v, s = self.initialize_adam()
        for iteration in range(max_iterations):
                print ("iteration : ", iteration)
                a, out, c, caches, pred = self.forward_pass(data)
                #error = 0.5 * np.power((out - self.y), 2)
                gradiants = self.backward_pass(self.da, caches)
                self.updateWeights(gradiants, v, s, t=1)
        print ("Training Results:")
        print (self.trainY)
        print (self.trainY.shape)
        print (pred)
        print (pred.shape)
        error = 0.5 * np.power((np.round(pred) - self.trainY), 2)
        print("After " + str(max_iterations) + " iterations, the total error for training is " + str(np.sum(error)))
        l = len(pred)
        acc = sum([np.round(pred[i]) == self.trainY[i] for i in range(l)]) / l
        print("Accuracy: %.2f%%" % (acc * 100))

    def predict(self, data, max_iterations=10, learning_rate=0.0005):

        a, out, c, caches, pred = self.forward_pass(data)
        print ("Test Results:")
        print (self.testy)
        print (self.testy.shape)
        print(pred)
        print(pred.shape)
        print (np.round(pred))
        error = 0.5 * np.power((np.round(pred) - self.testy), 2)
        print("After " + str(max_iterations) + " iterations, the total error for testing is " + str(np.sum(error)))
        l = len(pred)
        acc = sum([np.round(pred[i]) == self.testy[i] for i in range(l)]) / l
        print("Accuracy: %.2f%%" % (acc * 100))


if __name__ == "__main__":
    header = ['X1', 'X2', 'X3', 'X4', 'X5', "X6", "D"]
    dataurl = "train.csv"

    #neural_network = NeuralNet(dataurl, header)
    #neural_network.train(max_iterations=9000, learning_rate=0.02, act="sigmoid")
    #testError = neural_network.predict(act="sigmoid")
    #print("Final TestError is: ", testError)

    num_words = 500
    maxlen = 500
    (x_train1, y_train1), (x_test1, y_test1) = imdb.load_data(num_words = 500)
    x_train1 = sequence.pad_sequences(x_train1, maxlen=500)
    x_test1 = sequence.pad_sequences(x_test1, maxlen=500)
    print(y_train1.shape)
    x_train = x_train1

    y_train = y_train1
    e = Embedding(num_words, 5, input_length=maxlen)
    embeddingDim = 32
    model = Sequential()
    model.add(Embedding(num_words, embeddingDim, input_length = maxlen))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=2, batch_size=128, verbose=2)
    #scores = model.evaluate(x_train, y_train, verbose=0)
    #print(model.summary())
    #print("Accuracy: %.2f%%" % (scores[1] * 100))
    #print (model.predict(x_train))
    e = model.layers[0]
    weights = e.get_weights()[0]
    #print(weights)
    embeddings = model.layers[0].get_weights()[0]
    #words_embeddings = {w: embeddings[idx] for w, idx in word_to_index.items()}
    s1, s2 = x_train.shape
    XTrain = np.zeros((s1,s2,embeddingDim),dtype=np.float32)
    #print(embeddings.dtype)
    for i in range(0, s1):
        for j in range(0, s2):
            #print(embeddings[x_train[i,j]])
            XTrain [i,j] = embeddings[x_train[i,j]]
    XTrainRev = np.zeros((embeddingDim, s1,s2),dtype=np.float32)
    for i in range(0, s1):
        for j in range(0, s2):
            for k in range(0, embeddingDim):
                XTrainRev[k, i, j] = XTrain[i,j,k]

    #print(XTrainRev.shape)
    #print(XTrainRev)
    print("\n\n\n")
    #print(y_train)
    print("\n\n\n")
    #print(x_test1)
    print("\n\n\n")
    #print(y_test1)
#   embeded_train = np.asanyarray(output_array, dtype=np.float32)
    LSTMNetwork = RnnLSTM(XTrainRev, x_test1, y_train, y_test1, embeddingDim, model)
    LSTMNetwork.train(x_train,1, learning_rate = 0.02)
    LSTMNetwork.predict(x_test1,1, learning_rate=0.02)



