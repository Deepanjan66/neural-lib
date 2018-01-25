# Backpropagation
This is my implementation og a basic Neural Networks interface to show how the backpropagation algorithm works and how the structure of a network can influence the learning backpropagation facilitates.

To run the project:

1) Clone the repository
2) Create a virtual environment
3) Install all the dependencies by running `pip3 install -r requirements.txt`
4) `##cd src`
5) `##python3 run.py`

This will run the default run script. You can add your code and run the network.

## Neural Network utilities:

This implementation defines all the basic models you'd require for training 
sequential layers in neural networks. You can stack layers on top of each other 
and add activation functions to introduce non-linearity. Remember, networks created
from multiple layers will still be linear if no activation functions are added.

You can add your own activation functions in the `activations.py` file. There is
a base class that your class can implement. Please remember to implement all the
required methods.

You can also add your own error functions in the `error_evals.py` file. In the same
way, inherit from the base class and provide implementation of the required methods.

This implementation allows having different learning rates for different layers
in the network. Define a global learning rate by passing the `global_learning_rate`
aregument when you create the object graph and then you can change the learning
rate when you initialize the other layers. If a different learning rate is not
provided, the layers will use the global learning rate for backpropagation. 

All of the matrix operations in this implementation were performed using
python's highly optimised numpy library. The implementation uses
matrix operations for backpropagating the derivatives through the network.
You can print the matrices to see their weights as the network keeps on 
learning. 

You can define and shape of layers and any number of stacked layers as you like.
I will add skip connections in future iterations. This implementation should only be
used for learning and understanding how neural networks work. For real world applications,
please use packages such as tensorflow, keras, torch etc because they use advanced 
techniques for optiization and also support training with GPUs.

While learning with this implementation, if you think anything is not working, please do
not hesitate to post an issue here. I will try my best to rectify them as soon as possible.

Happy learning!
