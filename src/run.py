from models import *
from activations import *

def main():
    graph = Graph(3)
    graph.add(Layer(3, activation=Sigmoid()))
    graph.add(Layer(3, activation=Sigmoid()))
    graph.add(Layer(2, activation=Sigmoid()))
    for i in range(1000):
        graph.feed([1,1,1],[0,1])
    print(graph)

if __name__=="__main__":
    main()
