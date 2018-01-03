class NNError:
    """Have to convert this into an interface"""
    def eval_error(self, inputs, outputs):
        pass
    def derivative(self, output, target):
        pass

class SquaredSum(NNError):
    def eval_error(self, inputs, outputs):
        return sum([(inp-out)**2 for inp, out in zip(inputs, outputs)])
    def derivative(self, output, target):
        return output - target

