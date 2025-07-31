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

def encode(W, model):
    for w, p in zip(W, model.parameter()):
        p.data.copy_(w.reshape(p))

def get_depth_expansion_matrix(L1, L2):
    return nn.Parameter(torch.zeros(L2, L1))

def get_weight_dimension(W1, W2):
    L1, L2 = len(W1), len(W2)
    D_in, D_out, D1_in, D1_out, D2_in, D2_out = [], [], [], [], [], []
    D_in.append(W2[0].shape[1]); D_out.append(W2[0].shape[0])
    for l in range(1, L1-1):
        if l <= L2-2: # expansion, L2 > L1
            D_in.append(W2[l].shape[1]); D_out.append(W2[l].shape[0])
        else: # rarely use, contraction, L2 < L1
            D.append(len(W2[L2-2]))
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
        w1 = W1[l]
        print(f'{l=} {b.shape=} {w1.shape=} {a.shape=}')
        A.append(a); B.append(b)
    return A, B

def get_bias_width_expansion_matrices(W1, W2):
    raise NotImplementedError
