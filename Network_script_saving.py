# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:52:21 2022

@author: sahiz
"""

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import onnx
import torch.onnx as torch_onnx
from onnx_tf.backend import prepare

class nutANN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(nutANN, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H) #Declaration of layers
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, H)
        self.linear6 = torch.nn.Linear(H, H)
        self.linear7 = torch.nn.Linear(H, H)
        self.linear8 = torch.nn.Linear(H, D_out)
        self.dropout1 = torch.nn.Dropout(0.05)#Dropout p=0.1

        self.initialize_weight()
        
    def forward(self, x): #Definition of the forward pass
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h1 =torch.relu(self.dropout1(self.linear1(x)))
        h2 =torch.relu(self.linear2(h1))
        h3 =torch.relu(self.linear3(h2))
        h4 =torch.relu(self.linear4(h3))
        h5 =torch.relu(self.linear5(h4))
        h6 =torch.relu(self.linear6(h5))
        h7 =torch.tanh((self.linear7(h6)))
        alpha_pred =self.linear8(h7)    

        return alpha_pred
    
    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
 
def load_input(setNumber):
    Input=torch.load(f'D:/DNS_channel/trainingData/RANS_Correction/input_'+setNumber)
    beta=torch.load(f'D:/DNS_channel/trainingData/RANS_Correction/alpha_'+setNumber)
    
    return Input, beta

#%% Load network and example inputs

D_in, H, D_out =  8, 100, 1
N_batch=20
model = nutANN(D_in, H, D_out)

# choose a loss function and an optimizer
loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00008, max_lr=0.0008, step_size_up=50, cycle_momentum=False)
print("Network created!")

model.load_state_dict(torch.load('RANS_model/RANS_Deep_model_state_dict_epoch68'))
model.eval()
Input_val, beta_val = load_input('18')
Input_val=torch.tensor_split(Input_val, N_batch, dim=2)

alpha = model(Input_val[0])

# %% JIT compilation
traced_script_module = torch.jit.trace(model, Input_val[0])
traced_script_module.save('RANS_model/RANS_deep_model.pt')

# %% Torch to Onnx conversion
torch_onnx.export(model, Input_val[0], 'RANS_model/RANS_deep_model.onnx',
                  opset_version=12,       # Operator support version
                  input_names=['input'],   # Input tensor name (arbitary)
                  output_names=['output'])

print('Onnx model exported')

#%% Check of Onnx conversion
onnx_model = onnx.load('RANS_model/RANS_deep_model.onnx')  # load onnx model

# Check that the IR is well formed
onnx.checker.check_model(onnx_model)

# Print a Human readable representation of the graph
onnx.helper.printable_graph(onnx_model.graph)
print('Onnx model imported')
# onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'

#%% Onnx to TF conversion
# test = onnx_model.signatures["serving_default"]
device = 'cpu'

tf_rep = prepare(onnx_model,device)

print("Preparation OK!")

# print(tf_rep.inputs)
# print(tf_rep.outputs)
# print(tf_rep.tensor_dict)
tf_rep.export_graph('RANS_model/RANS_deep_model.pb')


print("Tensorflow Export OK!")

#%% Display of TF attributes
# import tensorflow as tf

# tf_model = tf.saved_model.load('tf_saved_model')
# tf_model.trainable = False

# input_tensor = tf.convert_to_tensor(Input_val[0].numpy())
# out = tf_model(**{'input': input_tensor})

import tensorflow as tf
from tensorflow.python.platform import gfile
GRAPH_PB_PATH = 'RANS_model/RANS_deep_model.pb'
with tf.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)
   print(names)
   
def print_inputs(pb_filepath):
    with tf.gfile.GFile(pb_filepath, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        input_list = []
        for op in graph.get_operations(): # tensorflow.python.framework.ops.Operation
            if op.type == "Placeholder":
                input_list.append(op.name)

        print('Inputs:', input_list)


def print_op(pb_filepath):
    with tf.gfile.GFile(pb_filepath, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        plist = []
        for op in graph.get_operations(): # tensorflow.python.framework.ops.Operation
            plist.append(op.name)
            print(op.values())
        print('Operations:', plist)



def print_outputs(pb_filepath):
    with open(pb_filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        name_list = []
        input_list = []
        for node in graph_def.node: # tensorflow.core.framework.node_def_pb2.NodeDef
            name_list.append(node.name)
            input_list.extend(node.input)

        outputs = set(name_list) - set(input_list)
        print('Outputs:', list(outputs))


print_inputs(GRAPH_PB_PATH)
print_outputs(GRAPH_PB_PATH)
print_op(GRAPH_PB_PATH)
