import numpy as np

class Neuron:
    def __init__(self, weights, bias, less_than_or_equals = False):
        self.weights = weights
        self.bias = bias
        self.less_than_or_equals = less_than_or_equals

    def classify(self, inputs):
        sum = 0
        for weight, input in zip(self.weights, inputs):
            sum += input * weight

        if self.less_than_or_equals:
            return 1 if sum <= self.bias else 0
        else:
            return 1 if sum >= self.bias else 0

def main():

    NOR_gate = Neuron([0.5, 0.5], 0.33, True)
    print("NOR gate:")
    for x1 in range(0,2):
        for x2 in range(0,2):
            print(f"{x1}, {x2}: {NOR_gate.classify([x1, x2])}")

    OR_gate = Neuron([0.5, 0.5], 0.33, False)
    NAND_gate = Neuron([0.5, 0.5], 0.66, True)
    AND_gate = Neuron([0.5, 0.5], 0.66, False)

    print("\nXOR gate:")
    for x1 in range(0,2):
        for x2 in range(0,2):
            print(f"{x1}, {x2}, {AND_gate.classify([OR_gate.classify([x1, x2]), NAND_gate.classify([x1, x2])])}")
    
    print("\nADDER gate:")
    for x1 in range(0,2):
        for x2 in range(0,2):
            out1  = NAND_gate.classify([x1, x2])
            out2  = NAND_gate.classify([x1, out1])
            out3  = NAND_gate.classify([out1, x2])
            sum   = NAND_gate.classify([out2, out3])
            carry = NAND_gate.classify([out1, out1])
            print(f"{x1}, {x2}, {carry}, {sum}")


if __name__ == "__main__":
    main()