import numpy as np
import random

class Neuron:
    def __init__(self, weights, bias, less_than_or_equals = False):
        self.weights = weights
        self.bias = bias
        self.delta = 0
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
        return f"Neuron Wn: {len(self.weights)} B: {self.bias}"

class NeuralNetwork:
    def __init__(self, num_layers: int, layer_depths: list) -> None:
        """_summary_

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
                    self.matrix[i].append(Neuron([random.uniform(-1, 1) for j in range(layer_depths[i])], random.uniform(-1, 1)))
                else: 
                    self.matrix[i].append(Neuron([random.uniform(-1, 1) for j in range(layer_depths[i-1])], random.uniform(-1, 1)))
        
        [[print(neuron) for neuron in layer] and print() for layer in self.matrix]

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

    NeuralNetwork(6, [2, 2, 3, 4, 2, 2])

if __name__ == "__main__":
    main()