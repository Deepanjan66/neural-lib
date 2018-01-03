from models import *
from activations import *

def main():
    graph = Graph(2)
    graph.add(Layer(3), activation=Sigmoid())
    graph.feed([1,1])

if __name__=="__main__":
    main()
