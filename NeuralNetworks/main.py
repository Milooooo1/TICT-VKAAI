import numpy as np
import random

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

    def update(self, delta, bias, weights):
        self.bias = bias
        self.delta = delta
        self.weights = weights

    def classify(self, inputs):
        sum = 0
        for weight, input in zip(self.weights, inputs):
            sum += input * weight

        if self.less_than_or_equals:
            return 1 if sum <= self.bias else 0
        else:
            return 1 if sum >= self.bias else 0

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
        
        [(print(f"Layer {index} with depth: {layer_depths[index]}:"), [print(neuron) for neuron in layer], print()) for index, layer in enumerate(self.matrix)]

    def feedForward(self):
        for index, layer in enumerate(self.matrix):
            if index == 0:
                continue #skip input layer
            
            for neuron in layer:
                acc = 0
                print(f"(1) zj = {round(neuron.bias, 2)} ", end="")
                for depth, weight in enumerate(neuron.weights):
                    acc += weight * self.matrix[index-1][depth].output
                    print(f"+ ({round(weight, 2)} * {round(self.matrix[index-1][depth].output, 2)})", end=" ")
                neuron.sum = neuron.bias + acc
                neuron.output = sigmoid(neuron.sum)
                print(f"= {round(neuron.bias + acc, 2)} || aj = {round(neuron.output, 2)}")
            
    def backPropagate(self, ground_truths):
        # First calculate the output layer
        print(f"\nDelta = r'(zi) * (ground truth - output)")
        for neuron, ground_truth in zip(self.matrix[-1], ground_truths):
            neuron.delta = sigmoid_grad(neuron.sum) * (ground_truth - neuron.output)
            print(f"(2) D = {round(sigmoid_grad(neuron.sum), 2)} * {round(ground_truth, 2)} - {round(neuron.output, 2)} = {round(neuron.delta, 2)}")
        
        # Backpropagate the previous layers
        print(f"\nDelta = r'(zi) * (deltaj * Weightij + ...)")
        for index, layer in enumerate(reversed((self.matrix[1:-1]))):
            for depth, neuron in enumerate(layer):
                acc = 0
                print(f"(3) D = {round(sigmoid_grad(neuron.sum), 2)} * ", end="")
                for i, prevNeuron in enumerate(self.matrix[index-1]):
                    acc += prevNeuron.weights[depth] * prevNeuron.delta
                    print(f"({round(prevNeuron.weights[depth], 2)} * {round(prevNeuron.delta, 2)})", end=" ")
                    if i != len(self.matrix[index-1])-1:
                        print(f"+", end=" ")
                neuron.delta = sigmoid_grad(neuron.sum) * acc
                print(f"= {round(sigmoid_grad(neuron.sum),2)} * {round(acc,2)} = {round(neuron.delta,2)}")
    
    def train(self, inputs, outputs, lr = 0.01):
        
        for input, neuron in zip(inputs, self.matrix[0]):
            neuron.output = input    
            
        self.feedForward()
        self.backPropagate(outputs)
        # self.updateWeights()
        
        print()
        for output, neuron in zip(outputs, self.matrix[1]):
            print(f"Neuron output: {neuron.output} vs ground truth: {output}")

def main():
    random.seed(0)

    # NOR_gate = Neuron([0.5, 0.5], 0.33, True)
    # print("NOR gate:")
    # for x1 in range(0,2):
    #     for x2 in range(0,2):
    #         print(f"{x1}, {x2}: {NOR_gate.classify([x1, x2])}")

    # OR_gate = Neuron([0.5, 0.5], 0.33, False)
    # NAND_gate = Neuron([0.5, 0.5], 0.66, True)
    # AND_gate = Neuron([0.5, 0.5], 0.66, False)

    # print("\nXOR gate:")
    # for x1 in range(0,2):
    #     for x2 in range(0,2):
    #         print(f"{x1}, {x2}, {AND_gate.classify([OR_gate.classify([x1, x2]), NAND_gate.classify([x1, x2])])}")
    
    # print("\nADDER gate:")
    # for x1 in range(0,2):
    #     for x2 in range(0,2):
    #         out1  = NAND_gate.classify([x1, x2])
    #         out2  = NAND_gate.classify([x1, out1])
    #         out3  = NAND_gate.classify([out1, x2])
    #         sum   = NAND_gate.classify([out2, out3])
    #         carry = NAND_gate.classify([out1, out1])
    #         print(f"{x1}, {x2}, {carry}, {sum}")

    nn = NeuralNetwork(3, [2, 2, 1])
    nn.train([1, 0], [1])

if __name__ == "__main__":
    main()