import torch.nn as nn
import numpy as np
import torch
import re

def get_parameter_by_name(model, parameter_name, get_module=False):
    if get_module:
        # Return the module instead of parameter
        for name, module in model.named_modules():
            if name == parameter_name:
                return module
        return None
    else:
        # Return the parameter as before
        for name, param in model.named_parameters():
            if name == parameter_name:
                return param
        return None

def get_model_size(model):
    return sum(p.numel() for p in model.parameters())

def decode(model):
    # initialize
    W, B = [], [] # weights, biases
    # loop throught and extract flatten weight
    for name, _ in model.named_parameters():
        if 'fc' in name:
            if 'weight' in name:
                w = get_parameter_by_name(model, name).detach()
                b = get_parameter_by_name(model, name.replace('weight', 'bias')).flatten().detach()
                W.append(w)
                B.append(b)
        else:
            raise NotImplementedError
    return W, B

def encode(W, B, model):
    print(f'[+] encoding model:')
    for name, module in model.named_modules():
        for i, layer in enumerate(module.children()):
            # print(name, layer, layer.weight.shape, W[i].shape, layer.bias.shape, B[i].shape)
            layer.weight = torch.nn.Parameter(W[i])
            layer.bias = torch.nn.Parameter(B[i])
    return model

def forward(x, W, B, model):
    # TODO: follow model definition more
    L = len(W)
    x = x.reshape(x.shape[0], -1)
    for l in range(L-1):
        x = torch.nn.functional.relu(x @ W[l].T + B[l])
    x = x @ W[L-1].T + B[L-1]
    return x

def get_depth_expansion_matrix(L1, L2):
    return nn.Parameter(torch.rand(L2, L1) / L1)

def get_weight_dimension(W1, W2):
    L1, L2 = len(W1), len(W2)
    D_in, D_out, D1_in, D1_out, D2_in, D2_out = [], [], [], [], [], []
    D_in.append(W2[0].shape[1]); D_out.append(W2[0].shape[0])
    for l in range(1, L1-1):
        if l <= L2-2: # expansion, L2 > L1
            D_in.append(W2[l].shape[1]); D_out.append(W2[l].shape[0])
        else: # rarely use, contraction, L2 < L1
            D_in.append(W2[L2-2].shape[1]); D_out.append(W2[L2-2].shape[0])
    D_in.append(W2[-1].shape[1]); D_out.append(W2[-1].shape[0])
    D1_in, D1_out = [_.shape[1] for _ in W1], [_.shape[0] for _ in W1]
    D2_in, D2_out = [_.shape[1] for _ in W2], [_.shape[0] for _ in W2]
    return D1_in, D1_out, D_in, D_out, D2_in, D2_out

def get_weight_width_expansion_matrices(W1, W2):
    D1_in, D1_out, D_in, D_out, D2_in, D2_out = get_weight_dimension(W1, W2)
    A, B = [], []
    L1, L2 = len(W1), len(W2)
    for l in range(L1):
        a = nn.Parameter(torch.zeros(D_in[l], D1_in[l]))
        b = nn.Parameter(torch.zeros(D_out[l], D1_out[l]))
        A.append(a); B.append(b)
    return A, B

def get_bias_dimension(B1, B2):
    L1, L2 = len(B1), len(B2)
    D, D1, D2 = [], [], []
    D.append(B2[0].shape[0])
    for l in range(1, L1-1):
        if l <= L2-2: # expansion, L2 > L1
            D.append(B2[l].shape[0])
        else: # rarely use, contraction, L2 < L1
            D.append(B2[L2-2].shape[0])
    D.append(B2[-1].shape[0])
    D1 = [_.shape[0] for _ in B1]
    D2 = [_.shape[0] for _ in B2]
    return D1, D, D2

def get_bias_width_expansion_matrices(B1, B2):
    D1, D, D2 = get_bias_dimension(B1, B2)
    B = []
    L1, L2 = len(B1), len(B2)
    for l in range(L1):
        b = nn.Parameter(torch.zeros(D[l], D1[l]))
        B.append(b)
    return B
