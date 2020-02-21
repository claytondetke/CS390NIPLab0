
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)

# Disable some troublesome logging.
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        f = self.__sigmoid(x)
        return f * (1 - f)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 2, minibatches = True, mbs = 100):
        #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        x = []
        for j in range(0, int(xVals.size/IMAGE_SIZE)):
            x.append(xVals[j].flatten())
        x = np.array(x)
        if(minibatches):
            Update1 = self.W1
            Update2 = self.W2
        for runs in range(0, epochs):
            print("epoch: %s" % str(runs))
            # Note: I was told by Sri that the better way to handle this was to use array math instead
            # of performing the operations on each individual value of x and y, and although slower, this works.
            for i in range(0, int(x.size/IMAGE_SIZE)):
                # Handle "minibatches" here, update weights after mbs iterations
                if(minibatches and i % mbs == 0):
                    print(i)
                    self.W1 = Update1
                    self.W2 = Update2
                l1, l2 = self.__forward(x[i])
                # Backprop
                l2d = (yVals[i] - l2) * self.__sigmoidDerivative(l2)
                l1d = np.dot(l2d, np.transpose(self.W2)) * self.__sigmoidDerivative(l1)
                if(minibatches):
                    Update2 += self.lr * np.transpose(np.dot(np.transpose(np.column_stack(l2d)), np.column_stack(l1)))
                    Update1 += self.lr * np.transpose(np.dot(np.transpose(np.column_stack(l1d)), np.column_stack(x[i])))
                else:
                    self.W2 += self.lr * np.transpose(np.dot(np.transpose(np.column_stack(l2d)), np.column_stack(l1)))
                    self.W1 += self.lr * np.transpose(np.dot(np.transpose(np.column_stack(l1d)), np.column_stack(x[i])))


    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2


# Build tensorflow model
def buildTFModel():
    model = keras.models.Sequential([keras.layers.Flatten(), keras.layers.Dense(128, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax)])
    lossType = keras.losses.categorical_crossentropy
    model.compile(optimizer = 'adam', loss = lossType, metrics=['accuracy'])
    return model

def trainTFModel(model, x, y, eps = 10):
    model.fit(x, y, epochs = eps)
    return model

# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

# Custom classifier that tries to determine the class label.
def customNetClassifier(xTest, model):
    ans = []
    for entry in xTest:
        pred = model.predict(entry.flatten())
        predB = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        predB[np.argmax(pred)] = 1
        ans.append(predB)
    return np.array(ans)

# Tensorflow classifier that tries to determine the class label.
def tfNetClassifier(xTest, model):
    pred = model.predict(xTest)
    ans = []
    for entry in pred:
        predB = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        predB[np.argmax(entry)] = 1
        ans.append(predB)
    return np.array(ans)

#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    #Added range reduction here (0-255 ==> 0.0-1.0).
    xTrain = tf.keras.utils.normalize(xTrain, axis=1)
    xTest = tf.keras.utils.normalize(xTest, axis=1)
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        model = NeuralNetwork_2Layer(IMAGE_SIZE, NUM_CLASSES, 128)
        model.train(xTrain, yTrain)
        return model
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = buildTFModel()
        model = trainTFModel(model, xTrain, yTrain)
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return customNetClassifier(data, model)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        return tfNetClassifier(data, model)
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    cm = np.zeros((10, 10))
    for i in range(preds.shape[0]):
        cm[np.argmax(yTest[i])][np.argmax(preds[i])] += 1
        if np.array_equal(preds[i], yTest[i]):
            acc = acc + 1
    accuracy = acc / preds.shape[0]
    precision = 0
    recall = 0
    for i in range(0, 10):
        precision += (cm[i][i] / np.sum(cm, axis = 0)[i]) / 10
        recall += (cm[i][i] / np.sum(cm, axis = 1)[i]) / 10
    f1 = 2 * precision * recall / (precision + recall)
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("Classifier precision: %f%%" % (precision * 100))
    print("Classifier recall: %f%%" % (recall * 100))
    print("Classifier F1 score: %f%%" % (f1 * 100))
    #print("Confusion matrix: ")
    #print(cm)
    print()


#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
