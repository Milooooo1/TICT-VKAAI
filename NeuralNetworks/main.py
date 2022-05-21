import numpy as np
import random
from sklearn.utils import shuffle
import csv

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_grad(x):
    return np.exp(-x)/(np.exp(-x)+1)**2

class Neuron:
    def __init__(self, weights, bias, less_than_or_equals = False):
        self.weights = weights
        self.bias = bias
        self.delta = 0
        self.output = 0
        self.less_than_or_equals = less_than_or_equals

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
        """_summary_

        Args:
            inputs (_type_): _description_
            outputs (_type_): _description_
            epochs (int, optional): _description_. Defaults to 1.
            lr (float, optional): _description_. Defaults to 0.01.
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

    def print(self):
        print()
        [(print(f"Layer {index} with depth: {len(self.matrix[index])}:"), [print(neuron) for neuron in layer], print()) for index, layer in enumerate(self.matrix[1:])]

def main():
    random.seed(0)
    
    xor_network = NeuralNetwork(3, [2, 2, 1])
    inputs = [[0,0],
              [1,0],
              [0,1],
              [1,1]]
    outputs = [[0], 
               [1], 
               [1], 
               [0]]
    
    print("XOR Network:")
    xor_network.train(inputs, outputs, 36376, 0.01)
    print()

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
        
    print("Iris Network:")
    iris_network = NeuralNetwork(4, [5, 5, 3, 3])
    iris_network.train(data[:100], labels[:100], 10000, 0.01)
    
    # TODO: Go through test set and measure accuracy

if __name__ == "__main__":
    main()
    print()