import numpy as np
import random
from sklearn.utils import shuffle
from tqdm import tqdm
import csv

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_grad(x):
    return np.exp(-x)/(np.exp(-x)+1)**2

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.delta = 0
        self.output = 0

    def __str__(self):
        '''Prints the number of weights and the bias'''
        return f"Neuron Wn: {[round(weight, 2) for weight in self.weights]} B: {round(self.bias, 2)}"

class NeuralNetwork:
    def __init__(self, num_layers: int, layer_depths: list) -> None:
        """The __init__ function of the NeuralNetwork class creates the network from the given dimensions
           It creates all the neurons for a layer and handles the first layer as an input layer from then
           on it creates neurons with the amount of weights as neurons in the previous layer.

        Args:
            num_layers (int): The amount of layers that need to be generated
            layer_depths (list): The depths of each of these layers

        Raises:
            IndexError: If not all the layers are matched with a depth an IndexError is raised 
                        because it is not know how many neurons to generate for this layer
        """
        if num_layers != len(layer_depths):
            raise IndexError("layer_depths need the same amount of layers as specified in num_layers")
        
        self.matrix = [[] for n in range(0, num_layers)]
        for i in range(0, num_layers):
            for j in range(0, layer_depths[i]):
                if i == 0:
                    self.matrix[i].append(Neuron([0 for j in range(layer_depths[i])], 0))
                else: 
                    self.matrix[i].append(Neuron([random.uniform(-1, 1) for j in range(layer_depths[i-1])], random.uniform(-1, 1)))

    def feedForward(self):
        """calculate the outputs of all the neurons in the network.
        """
        for index, layer in enumerate(self.matrix):
            if index == 0:
                continue #skip input layer
            
            for neuron in layer:
                acc = 0
                # print(f"(1) zj = {round(neuron.bias, 2)} ", end="")
                for depth, weight in enumerate(neuron.weights):
                    acc += weight * self.matrix[index-1][depth].output
                    # print(f"+ ({round(weight, 2)} * {round(self.matrix[index-1][depth].output, 2)})", end=" ")
                neuron.sum = neuron.bias + acc
                neuron.output = sigmoid(neuron.sum)
                # print(f"= {round(neuron.bias + acc, 2)} || aj = {round(neuron.output, 2)}")
            
    def backPropagate(self, ground_truths: list):
        """calculate all the delta's for the whole network.

        Args:
            ground_truths (list): the known outputs that the network needs to match
        """
        # First calculate the output layer
        # print(f"\nDelta = r'(zi) * (ground truth - output)")
        for neuron, ground_truth in zip(self.matrix[-1], ground_truths):
            neuron.delta = sigmoid_grad(neuron.sum) * (ground_truth - neuron.output)
            # print(f"(2) D = {round(sigmoid_grad(neuron.sum), 2)} * {round(ground_truth, 2)} - {round(neuron.output, 2)} = {round(neuron.delta, 2)}")
        
        # Backpropagate the previous layers
        # print(f"\nDelta = r'(zi) * (deltaj * Weightij + ...)")
        for index, layer in enumerate(reversed((self.matrix[1:-1]))):
            for depth, neuron in enumerate(layer):
                acc = 0
                # print(f"(3) D = {round(sigmoid_grad(neuron.sum), 2)} * ", end="")
                for i, prevNeuron in enumerate(self.matrix[index-1]):
                    acc += prevNeuron.weights[depth] * prevNeuron.delta
                    # print(f"({round(prevNeuron.weights[depth], 2)} * {round(prevNeuron.delta, 2)})", end=" ")
                    # if i != len(self.matrix[index-1])-1:
                        # print(f"+", end=" ")
                neuron.delta = sigmoid_grad(neuron.sum) * acc
                # print(f"= {round(sigmoid_grad(neuron.sum),2)} * {round(acc,2)} = {round(neuron.delta,2)}")
    
    def updateWeights(self, learning_rate: int):
        """Update the weights and biases of each neuron.

        Args:
            learning_rate (int): How fast the network should learn, bigger learning rate is faster
                                 but can cause a less effective network. A lower learning rate is slower
                                 and results in a better network but can stop to early if its stuck.
        """
        # print()
        for index, layer in enumerate(reversed(self.matrix[1:])):
            for neuron in layer:
                for depth, weight in enumerate(neuron.weights):
                    tmp = len(self.matrix) - 1
                    # print(f"(4) {round(weight,2)} += {learning_rate} * {round(neuron.delta, 2)} * {round(self.matrix[tmp-index-1][depth].output, 2)} = {round(neuron.weights[depth] + (learning_rate * neuron.delta * self.matrix[tmp-index-1][depth].output),2)}")     
                    # print(f"(4) {round(neuron.bias, 2)} += {learning_rate} * {round(neuron.delta, 2)} = {round(neuron.bias + (learning_rate * neuron.delta), 2)}")     
                    neuron.weights[depth] += learning_rate * neuron.delta * self.matrix[tmp-index-1][depth].output
                    neuron.bias += learning_rate * neuron.delta
    
    def train(self, inputs: list, outputs: list, epochs: int = 1, lr: int = 0.01):
        """The train function trains the network by iterating over the dataset and calling
           all the specific calculation functions. It also keeps track of the accuracy. 

        Args:
            inputs (list): list of all input parameters
            outputs (list): list of expected outputs
            epochs (int, optional): the amount of times the 
                                    train function should iterate
                                    over the dataset. Defaults to 1.
            lr (float, optional): how fast the AI should learn, 
                                  bigger is faster but not always better. 
                                  Defaults to 0.01.
        """
        
        for epoch in range(1, epochs+1):
            correct_ctr = 0
            loss = 0
            for index, data_point in enumerate(inputs):
                # self.print()
                
                for input, neuron in zip(data_point, self.matrix[0]):
                    neuron.output = float(input)    
                    
                self.feedForward()
                self.backPropagate(outputs[index])
                self.updateWeights(lr)
                
                # print(outputs[index].index(max(outputs[index])))
                network_outputs = [neuron.output for neuron in self.matrix[-1]]
                # print(network_outputs.index(max(network_outputs)))
                if outputs[index].index(max(outputs[index])) == network_outputs.index(max(network_outputs)):
                    correct_ctr += 1
                for output, neuron in zip(outputs[index], self.matrix[-1]):
                    # print(f"Neuron output: {neuron.output} vs ground truth: {output}")
                    loss += (output - neuron.output)
                    # if (round(neuron.output) == output): 
                        # correct_ctr+=1
                    
                # self.print()
            print(f"Epoch: {epoch}/{epochs} | {round(correct_ctr)}/{len(inputs)} correct | accuracy of: {round((correct_ctr) / len(inputs)*100)}% | loss: {round(loss/len(inputs), 8)} | [{round((epoch/epochs)*100,2)}%] completed      " , end="\r")
        print()

    def test(self, inputs: list, outputs: list):
        """The test function goes through the list of inputs and forwards the inputs
           to measure the accuracy the outputs after forwarding are compared to the ground truths

        Args:
            inputs (list): list of all input parameters
            outputs (list): list of expected outputs
        """
        correct_ctr = 0
        for index, input in tqdm(enumerate(inputs), desc="Testing network"):

            for input, neuron in zip(input, self.matrix[0]):
                neuron.output = float(input)

            self.feedForward()

            network_outputs = [neuron.output for neuron in self.matrix[-1]]
            if outputs[index].index(max(outputs[index])) == network_outputs.index(max(network_outputs)):
                correct_ctr += 1   
        
        print(f"Accuracy: {round((correct_ctr/len(outputs))*100)}") 

    def classify(self, input: list):
        """The classify function classifies a single input and returns the estimated output 

        Args:
            input (list): list of all the inputs, 
                          must be same dimensions as input layer

        Returns:
            estimated outputs: list of all the estimated outputs
        """
        for input, neuron in zip(input, self.matrix[0]):
                neuron.output = float(input)
        self.feedForward()
        return [neuron.output for neuron in self.matrix[-1]]


    def __str__(self):
        return [(f"Layer {index} with depth: {len(self.matrix[index])}:", [str(neuron) for neuron in layer]) for index, layer in enumerate(self.matrix[1:])]

    def print(self):
        [(print(layer_depth), [print(neuron) for neuron in neurons], print()) for layer_depth, neurons in self.__str__()]


def main():
    random.seed(0)
    
    # The xor network has two outputs for each state, 1 or 0 
    # my accuracy detection is not smart so you need a output neuron for every possible state
    xor_network = NeuralNetwork(3, [2, 2, 2])
    inputs = [[0,0],
              [1,0],
              [0,1],
              [1,1]]
    outputs = [[1, 0], 
               [0, 1], 
               [0, 1], 
               [1, 0]]
    
    print("XOR Network:")
    xor_network.train(inputs, outputs, 9000, 0.05)
    xor_network.test(inputs, outputs)

    data = []
    labels = []
    with open('/home/milo/TICT-VKAAI/NeuralNetworks/iris.data', newline='') as csvfile:
        spamreader = np.array(list(csv.reader(csvfile, delimiter=',', quotechar='|')))
        data = spamreader[:, :-1]
        named_labels = spamreader[:, -1:]
        
        data, named_labels = shuffle(data, named_labels)
        for label in named_labels:
            if label == 'Iris-setosa': labels.append([1,0,0])
            elif label == 'Iris-versicolor': labels.append([0,1,0])
            elif label == 'Iris-virginica': labels.append([0,0,1])
        
    print("\nIris Network:")
    iris_network = NeuralNetwork(4, [5, 5, 3, 3])
    iris_network.train(data[:100], labels[:100], 9000, 0.01)
    iris_network.test(data[100:], labels[100:])
    

if __name__ == "__main__":
    main()