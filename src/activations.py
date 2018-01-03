import math

class Activation:
    def evaluate(self, inputs):
        pass
    def derivative(self, inp):
        pass

class Sigmoid(Activation):
    def evaluate(self, inp):
        return 1/(1+exp(-inp))
    
    def derivative(self, inp):
        return inp*(1-inp)

