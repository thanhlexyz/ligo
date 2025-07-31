import torch.optim as optim
import torch.nn as nn
import torch

from . import util

class Initializer:

    def __init__(self, args):
        pass

    def init(self, pretrain_model, model, loader):
        # step 1:
        # construct trainable depth expansion matrix
        L1 = L2 = 0
        theta1 = theta2 = [] # list of flatten parameters
        for name, _ in pretrain_model.named_parameters():
            if 'fc' in name and 'weight' in name:
                L1 += 1
                w = util.get_parameter_by_name(pretrain_model, name).flatten()
                b = util.get_parameter_by_name(pretrain_model, name.replace('weight', 'bias')).flatten()
                p = torch.cat([w, b])
                print(p.shape)
                theta1.append(p)
        exit()
        # step 2:
        # construct trainable width expansion matrix
        # step 3:
        # optimize expansion matrices
        print('[+] optimize LiGO weight transfer matrices')
        criterion = nn.MSELoss()
        optimizer = optim.Adam(..., lr=args.lr)
        for i, (x, _) in enumerate(loader):
            optimizer.zero_grad()
            y_hat_pretrain = pretrain_model(x) # may be need to modify here
            y_hat = model(x)
            loss = criterion(y_hat_pretrain, y_hat)
            loss.backward()
            optimizer.step()
            print(f'    - {i=} {loss.item()=:0.6f}')
        # step 4:
        # copy weight to model for return
        return model
