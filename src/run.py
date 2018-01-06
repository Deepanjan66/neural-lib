from models import *
from activations import *

def main():
    graph = Graph(2, global_learning_rate=0.3)
    graph.add(Layer(2, activation=Sigmoid()))
    graph.add(Layer(1))
    for i in range(100000):
        graph.feed([[1,1],[1,0]],[[0.3],[0.2]])

if __name__=="__main__":
    main()
