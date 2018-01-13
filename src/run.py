from models import *
from activations import *

def main():
    # Create graph. The number of input nodes is a required argument
    graph = Graph(2, global_learning_rate=0.3)
    # Add a new layer of 2 neurons and sigmoid activation
    graph.add(Layer(2, activation=Sigmoid()))\
    # Add a new layer of one neuron and no activation function
    graph.add(Layer(1))
    # Train network for a 1000 iterations and check outputs
    for i in range(1000):
        graph.feed([[1,1],[1,0]],[[0.3],[0.2]])
    
    # Predict output for input.
    # Network is overfitting here but this
    # has been shown just for demonstration.
    print(graph.predict([[1,1]]))

if __name__=="__main__":
    main()
