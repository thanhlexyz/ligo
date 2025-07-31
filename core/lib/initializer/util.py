import torch.nn as nn
import torch

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

def decode(model):
    # initialize
    W = [] # weights
    # loop throught and extract flatten weight
    for name, _ in model.named_parameters():
        if 'fc' in name:
            if 'weight' in name:
                w = get_parameter_by_name(model, name).flatten().detach()
                b = get_parameter_by_name(model, name.replace('weight', 'bias')).flatten().detach()
                W.append(torch.cat([w, b]))
        else:
            raise NotImplementedError
    return W, len(W)

def encode(W, model):
    for w, p in zip(W, model.parameter()):
        p.data.copy_(w.reshape(p))

def get_depth_expansion_matrix(L1, L2):
    return nn.Parameter(torch.zeros(L2, L1))

def get_dimension(W1, W2, L1, L2):
    D, D1, D2 = [], [], []
    D.append(len(W2[0]))
    for l in range(1, L1-1):
        if l <= L2-2: # expansion, L2 > L1
            D.append(len(W2[l]))
        else: # rarely use, contraction, L2 < L1
            D.append(len(W2[L2-2]))
    D.append(len(W2[-1]))
    D1 = [len(_) for _ in W1]
    D2 = [len(_) for _ in W2]
    return D1, D, D2

def get_width_expansion_matrices(W1, W2, L1, L2):
    D1, D, D2 = get_dimension(W1, W2, L1, L2)
    A, B = [], []
    for d1, d in zip(D1, D):
        a = nn.Parameter(torch.zeros(d, d1))
        b = nn.Parameter(torch.zeros(d, d1))
        print(a.shape, b.shape)
        A.append(a); B.append(b)
    return A, B
