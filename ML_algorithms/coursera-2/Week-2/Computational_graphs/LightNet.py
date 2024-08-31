"""
LightNet is a very subtle attempt at implementing TensorFlow/PyTorch like neural networks which use Rev_autodiff
and computational graphs. LightNet is only for educational purposes and can be used to implement and train nets
over small and moderately large datasets (maxsize after tiling can be of the order 10^6) in a reasonable time frame.

To dive deep into how LightNet was built refer the Jupyter Notebook 'Reverse_autodiff_py.ipynb'.
"""

import numpy as np
import graphviz

class Node:
    def __init__(self, data, label='', children=(), _op=[]):
        self.data=np.array(data, dtype=float)
        self.label=label
        self.children=children
        self._op=_op    # basically each node itself stores the relation which relates it to its children in _op
        self._grad=np.zeros(shape=self.data.shape, dtype=float)   # stores the derivative of the root node w.r.t the current node
        # each element of _grad represents a seperate parameter
        # Thus each element of _grad will store the derivative of root node w.r.t that parameter
        self._backward= lambda: None
    
    # A magic function which shows customised msg to display the object
    def __repr__(self):
        return f"Node: data={np.array2string(self.data, precision=4, floatmode='fixed')}, label='{self.label}', op='{self._op}'"
    
    def __add__(self, other):
        new_node=Node(self.data+other.data, children=(self, other), _op=["+"])
        
        # Defining the backward function for the new_node
        # will do 2 things:
        # 1) compute the gradients of new_node's children w.r.t the root using the chain rule
        # 2) call the backward function for new_node's children if they are not leaf nodes
        def _backward():
            new_node.children[0]._grad+=new_node._grad
            new_node.children[1]._grad+=new_node._grad
            
            for child in new_node.children:
                if(child._op):  #child is not a leaf node
                    child._backward()
        
        # Any Node which is formed by the addition operation, will know through _backward() what to do in reverse pass.
        # here a function object of the _backward() function is passed to the new_node object's _backward attribute
        # we need to take speacial care while considering the scope of the _backward() fucntion object.
        new_node._backward=_backward
            
        return new_node
    
    # this operation multiplies corresponding elements of self.data and other.data and returns a Node obejct with data as 
    # array of same size.
    def __mul__(self, other):
        new_node=Node(self.data*other.data, children=(self, other), _op=['*'])
        
        def _backward():
            new_node.children[0]._grad+=(new_node._grad)*(new_node.children[1].data)
            new_node.children[1]._grad+=(new_node._grad)*(new_node.children[0].data)
            
            for child in new_node.children:
                if(child._op):  #child is not a leaf node
                    child._backward()
                    
        new_node._backward=_backward
        return new_node
    
    # creates a Node with data as the dot product of two vectors
    def dotpr(self, other):
        new_node=Node(np.sum((self.data)*(other.data)), children=(self, other), _op=['dot'])
        
        def _backward():
            new_node.children[0]._grad+=(new_node._grad)*(new_node.children[1].data)
            new_node.children[1]._grad+=(new_node._grad)*(new_node.children[0].data)
            new_node.children[0]._backward()
            new_node.children[1]._backward()
            
        new_node._backward=_backward
        return new_node
    
    def sig(self):
        new_node=Node(1/(1+pow(np.e, -self.data)), children=(self, ),_op=['sig'])
        
        def _backward():
            new_node.children[0]._grad+=(new_node.data)*(1-new_node.data)*new_node._grad
            
            if(new_node.children[0]._op):  #child is not a leaf node
                new_node.children[0]._backward()
                    
        new_node._backward=_backward
        return new_node
    
    
    # The loss function
    def BinCrossEntropy(self, y):   # y is the target value for a given datapoint
        new_node=Node((-y*np.log(self.data)-(1-y)*np.log(1-self.data)),children=(self,), _op=["BCE"])

        def _backward():
            new_node.children[0]._grad+=(((1-y)/(1-new_node.children[0].data))-(y/new_node.children[0].data))*new_node._grad

            if(new_node.children[0]._op):  #child is not a leaf node
                new_node.children[0]._backward()

        new_node._backward=_backward
        return new_node
        
    # fucntion to convert a list of Nodes with monovalued data to a single Node with array data
    # used to connect 2 consecutive layers of a neural network
    @staticmethod
    def buffer(Nodes:list):
        new_node=Node([node.data for node in Nodes], children=tuple(Nodes), _op=['buf'])
        
        def _backward():
            i=0
            for child in new_node.children:
                child._grad+=new_node._grad[i]
                i+=1
            
            for child in new_node.children:
                if(child._op):
                    child._backward()
                    
        new_node._backward=_backward
        return new_node
        
    # Initialising the backward pass to compute gradients
    # The Node in single quotes while specifying type of root is used for fwd declaration in python
    @staticmethod
    def backward(root:'Node'):
        root._grad=np.ones(shape= root.data.shape, dtype=float)
        root._backward()
        
def draw_graph(root: Node)-> graphviz.graphs.Digraph:
    #initialising the graph
    graph=graphviz.Digraph(format='svg', name="Comp_graph", graph_attr={"rankdir":"LR"}, comment="Computational graph")
    
    # creating the parent node in the visualisation
    uidr=str(id(root))  # unique id of the root Node object
    graph.node(uidr, label=f"{root.label} | data: {np.array2string(root.data, precision=4, floatmode='fixed')} | grad: {np.array2string(root._grad, precision=4, floatmode='fixed')}", shape="record")
    
    if root._op:    #checking if root._op is empty or not, if not empty then create node for operation
        uidr_op=str(id(root._op))
        graph.node(uidr_op, label=f"{root._op}")
        graph.edge(uidr_op, uidr)   # uidr_op (tail) -> uidr (head)
    else:
        return graph
    
    """ there exists a possibilty that 2 Nodes share a single child, in such a scenario we dont want to create
    two nodes in the visualisation for a single Node object, thus we'll have to maintain a "SET" of Node objects
    which have been aldready included in the representation."""
    
    """
    There also exists a possibility that a Node calls its children again and again due to calls from multiple parents
    to the Node. We need to avoid multiple calls from a given parent while visualisaing the graph. (multiple recursive
    calls from the same parent to the same child lead to unecessary arrows in the graph visualisation).
    """
    check=set((uidr))
    edges={(uidr, uidr_op)}
    
    def rec(curr: Node, parent: Node):
        uid=str(id(curr))
        if uid not in check:   #checking if current node aldready has a node in visualisation or not
            # creating curr Node object in the visualisation
            graph.node(uid, label=f"{curr.label} | data: {np.array2string(curr.data, precision=4, floatmode='fixed')} | grad: {np.array2string(curr._grad, precision=4, floatmode='fixed')}", shape="record")
            check.add(uid)
        if tuple((uid, str(id(parent._op)))) in edges:
            return
        graph.edge(uid, str(id(parent._op)))    # add an edge between parent's operation and curr node
        edges.add((uid, str(id(parent))))
            
        # adding an edge between current node and its operation
        if curr._op:
            uid_op=str(id(curr._op))
            graph.node(uid_op, label=f"{curr._op}")
            if tuple((uid_op, uid)) in edges:
                return
            graph.edge(uid_op, uid)
            edges.add((uid_op,uid))
        else:   # if the current Node object has no operation i.e. it is a leaf node
            return
        
        for child1 in curr.children:
            rec(child1, curr)
        
    for child in root.children:
        rec(child, root)    # recursive call to immidiate children of the root node.
        
    return graph

class Neuron:
    def __init__(self, num_par, activation, label=''):
        self.label=label
        self.activation = activation
        self.num_par=num_par
        # rand_obj=np.random.RandomState(seed=123) was used earlier to keep getting same initial weights and troubleshoot
        self.weights=Node(np.random.randn(num_par+1), label=f"W: {self.label}")    # the last weight will be the bias
        # Inputs for an entire layer are the outputs of the previous layer
        # thus there is no specific need to create spereate Node objects for each neuron for the same input array 
        #self.inputs=Node(np.ones(num_par+1), label="")
        
    def __call__(self, input:Node):
        wdoti=self.weights.dotpr(input); wdoti.label=f"D: {self.label}"
        sigz=wdoti.sig(); sigz.label=f"S: {self.label}"
        self.wdoti=wdoti
        self.sigz=sigz
        return self.sigz
        
class Layer:
    def __init__(self, units:int, activation='',num_inp=0, label=''):
        self.units=units
        self.activation=activation
        self.num_inp=num_inp
        self.label=label
        self.neuron_list=[]
        
        # this will keep track of the input Node to a layer(used in weight update step to reset gradients)
        # this input node could be given by the user in the __call__ fucntion or can be buffer node storing previous
        # layer's outputs aswell
        
        # every layer will have to take some input inorder to compute further
        self.inputs=Node(np.zeros((self.num_inp+1)))
        if(num_inp!=0):     
            """
            IF A LAYER IS EXPLICITLY DECLARED i.e not through the network class.
            """
            self.neuron_list=[Neuron(self.num_inp,self.activation,label=f"{self.label}, N{i}") for i in range(units)]
    
    def create_neurons(self):
        """
        INCASE A LAYER IS IMPLICITLY DECLARED i.e. through the network class.
        """
        self.neuron_list=[Neuron(self.num_inp,self.activation,label=f"{self.label}, N{i}") for i in range(self.units)]
    
    #initiates a fwd pass through the layer
    def __call__(self, input:Node):
        self.inputs=input
        # This buffer Node connects the outputs of Neurons of a layer to the next layer.
        # These outputs serve as inputs to the next layer.
        buffer_node=Node.buffer([neuron(self.inputs) for neuron in self.neuron_list]); buffer_node.label=f"I: L{int(self.label[1])+1}"
        return buffer_node
    
    def get_weights(self):
        weights=np.array([neuron.weights.data for neuron in self.neuron_list])
        return weights
               
class Network:
    def __init__(self, layers:list):
        self.layers=layers
        self.inp_par=layers[0][0]   # the shape of the input is defined by the first element of the list passed
        
        # 1) INITIALISING NUM_INP FOR ALL LAYERS OF THE NETWORK
        layers[1].num_inp=self.inp_par   # the number of weights (excl. bias) each neuron of this layer would have
        layers[1].create_neurons()
        for i in range(2,len(layers)):
            layers[i].num_inp=layers[i-1].units            
            # for eg: num of units in the first layer=3; num of inputs for the next layer=3(excl. bias)
            # num of weights for each neuron in the second layer=3(excl. bias)
            
            # 2) CREATE NEURON OBJECTS FOR EACH LAYER
            layers[i].create_neurons()
            
        # 3) CONNECTING THE LAYERS: to connect the layers we'll have to fwd pass through the network once
        # initialising the root, the root Node has _op as buffer
        # The below line calls the __call__ fucntion of the newtwork being initialised.
        # this initialisation is important because connects all the layers of the network
        self.root=self(np.random.rand(self.inp_par))
            
        
    def __call__(self, input):
        input=np.array(input).reshape((self.inp_par))
        #adding 1 at the end of the input array
        input=np.r_[input, 1]    
        #creating input Node object, this object belongs to the entire layer
        output=Node(input,label=f"I: L1")
        for layer in self.layers[1:]:
            output=layer(output)
            """ 
            Consider an example, suppose the current layer has 3 output neurons, thus layer(temp) would return a Node
            object with data as a np.array of size (3,). But now Neurons of the next layer would need to have an input
            of size (4,). Thus, we'll have to add the bias term to 'output' Node.
            """
            output.data=np.r_[output.data, 1]
            output._grad=np.r_[output._grad, 0]
        output.data=np.array([output.data[0]], dtype=float)
        output._grad=np.array([output._grad[0]], dtype=float)
        self.root=output    # saving the root Node object of the Network
        return output
    
    def backward(self):
        Node.backward(self.root)
    
    def train(self, X, Y, epoch, alpha):
        #l=[]
        X=np.array(X, dtype=float)
        Y=np.array(Y, dtype=float)
        for _ in range(epoch):
            for i in range(X.shape[0]):
                x=X[i]
                y=Y[i]
                # 1) fwd pass and calculating loss for the datapoint assuming activation to be sigmoid
                y_p=self(x)
                self.root=y_p.BinCrossEntropy(y); self.root.label='Loss'
                print(self.root)
                # 2) backward pass
                self.backward()
                
                # 3) Weight update step: data=data-(alpha*_grad) for every neuron
                # only the weights of the neurons will be updated
                # once the wieghts are updated the _grad attribute will be reset to 0 for the next iteration
                # NOTE: that _grad attribute will have to be reset for all nodes not just the weights
                for layer in self.layers[1:]:
                    for neuron in layer.neuron_list:
                        neuron.weights.data = neuron.weights.data-(alpha*neuron.weights._grad)                        
                        neuron.weights._grad=np.zeros(neuron.weights.data.shape)
                        neuron.wdoti._grad=np.zeros(neuron.wdoti.data.shape)
                        neuron.sigz._grad=np.zeros(neuron.sigz.data.shape)
                    layer.inputs._grad=np.zeros(layer.inputs._grad.shape)   # Input Node objects of layers (_op= buffer)
                self.root.children[0]._grad=np.array([0], dtype=float)   # This is a Node object with _op = buffer
                self.root._grad=np.array([0], dtype=float)   # This is a Node object with _op = BCE
                
                # 'l' contains graphviz.Digraph objects, each is a graph made after one weight update step
                # l.append(self.draw())
        self.root=self.root.children[0]
        #return l
    
    def score(self, X,Y):
        i=0
        count=0
        for x in X:
            c=float(self(x).data[0])>.5
            if Y[i]==c:
                count+=1
            i+=1
        return count/Y.shape[0]             
    
    def get_weights(self):
        for layer in self.layers[1:]:
            print(f"{layer.label}\nWeights: \n{layer.get_weights()}")
        
    def draw(self):
        return draw_graph(self.root)