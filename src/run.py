from models import *

def main():
    graph = Graph(2)
    graph.add(Layer(3))
    graph.feed([1,1])

if __name__=="__main__":
    main()
